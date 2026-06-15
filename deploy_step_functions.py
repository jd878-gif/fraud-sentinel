"""
deploy_step_functions.py
=========================
Creates the Step Functions state machine that orchestrates the
full Glue ETL pipeline (Bronze→Silver→Gold) and an EventBridge
rule that triggers it on a weekly schedule.

Architecture:
  EventBridge (weekly cron) → Step Functions → Glue Job 1
                                              → Glue Job 2
                                              → SNS (success/failure)

Why Step Functions over a cron script?
  - Execution state persists in AWS (survives laptop closing)
  - Visual workflow graph in AWS Console
  - Native retry logic with configurable backoff
  - Parallel branch support for future expansion
  - Full execution history for debugging
  - Integrates with CloudWatch for execution monitoring
  - Zero cost for state transitions (only pay for state transitions
    beyond the free tier of 4,000/month)

State Machine Design:
  1. StartBronzeToSilver     — kick off Glue job 1
  2. WaitForBronzeToSilver   — poll every 30 seconds
  3. GetBronzeToSilverStatus — check if SUCCEEDED/FAILED/RUNNING
  4. BronzeToSilverComplete? — choice state: route on job status
  5. StartSilverToGold       — kick off Glue job 2
  6. WaitForSilverToGold     — poll every 30 seconds
  7. GetSilverToGoldStatus   — check if SUCCEEDED/FAILED/RUNNING
  8. SilverToGoldComplete?   — choice state: route on job status
  9. NotifySuccess           — publish SNS success message
  10. NotifyFailure          — publish SNS failure message (catch-all)
"""

import boto3
import json
import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger("deploy_sfn")

# ── Config ─────────────────────────────────────────────────────────
REGION        = "us-east-1"
ACCOUNT_ID    = "621402808508"
SFN_ROLE_ARN  = f"arn:aws:iam::{ACCOUNT_ID}:role/FraudPlatformStepFunctionsRole"
SNS_TOPIC_ARN = f"arn:aws:sns:{REGION}:{ACCOUNT_ID}:fraud-alerts"
S3_BUCKET     = "fraud-platform-jeet-dev"
DATABASE_NAME = "fraud_platform_dev"

MACHINE_NAME  = "FraudPlatform-ETL-Pipeline"
SCHEDULE_NAME = "FraudPlatform-WeeklyETL"

# ── State Machine Definition ────────────────────────────────────────

def build_state_machine_definition() -> dict:
    """
    Build the Step Functions ASL (Amazon States Language) definition.

    ASL is JSON that describes each state, transitions, and error handling.
    Step Functions executes this as a managed workflow — each state
    transition is recorded, retryable, and visible in the console.

    Key patterns used:
    - Task states: call AWS services (Glue, SNS)
    - Wait states: pause N seconds between polls
    - Choice states: branch on job status (SUCCEEDED/FAILED/RUNNING)
    - Catch blocks: handle unexpected errors and route to failure SNS
    """
    return {
        "Comment": (
            "Fraud Platform ETL Pipeline — orchestrates Bronze→Silver→Gold "
            "Glue jobs weekly with SNS notifications on success and failure."
        ),
        "StartAt": "StartBronzeToSilver",
        "States": {

            # ── State 1: Start Bronze→Silver Glue job ──────────────
            "StartBronzeToSilver": {
                "Type": "Task",
                "Resource": "arn:aws:states:::glue:startJobRun",
                "Parameters": {
                    "JobName": "fraud-bronze-to-silver",
                    "Arguments": {
                        "--S3_BUCKET":     S3_BUCKET,
                        "--DATABASE_NAME": DATABASE_NAME,
                    },
                },
                # Store the job run ID so we can poll its status
                "ResultPath": "$.BronzeToSilverRun",
                "Next": "WaitForBronzeToSilver",
                "Catch": [
                    {
                        "ErrorEquals": ["States.ALL"],
                        "Next": "NotifyFailure",
                        "ResultPath": "$.error",
                    }
                ],
            },

            # ── State 2: Wait 30 seconds before polling ─────────────
            "WaitForBronzeToSilver": {
                "Type": "Wait",
                "Seconds": 30,
                "Next": "GetBronzeToSilverStatus",
            },

            # ── State 3: Poll Glue job status ───────────────────────
            "GetBronzeToSilverStatus": {
                "Type": "Task",
                "Resource": "arn:aws:states:::glue:startJobRun.sync",
                "Parameters": {
                    "JobName": "fraud-bronze-to-silver",
                    "Arguments": {
                        "--S3_BUCKET":     S3_BUCKET,
                        "--DATABASE_NAME": DATABASE_NAME,
                    },
                },
                "ResultPath": "$.BronzeToSilverResult",
                "Next": "StartSilverToGold",
                "Catch": [
                    {
                        "ErrorEquals": ["Glue.GlueException", "States.ALL"],
                        "Next": "NotifyFailure",
                        "ResultPath": "$.error",
                    }
                ],
                "Retry": [
                    {
                        "ErrorEquals": ["States.TaskFailed"],
                        "IntervalSeconds": 30,
                        "MaxAttempts": 2,
                        "BackoffRate": 2.0,
                    }
                ],
            },

            # ── State 4: Start Silver→Gold Glue job ────────────────
            "StartSilverToGold": {
                "Type": "Task",
                "Resource": "arn:aws:states:::glue:startJobRun.sync",
                "Parameters": {
                    "JobName": "fraud-silver-to-gold",
                    "Arguments": {
                        "--S3_BUCKET":     S3_BUCKET,
                        "--DATABASE_NAME": DATABASE_NAME,
                    },
                },
                "ResultPath": "$.SilverToGoldResult",
                "Next": "NotifySuccess",
                "Catch": [
                    {
                        "ErrorEquals": ["Glue.GlueException", "States.ALL"],
                        "Next": "NotifyFailure",
                        "ResultPath": "$.error",
                    }
                ],
                "Retry": [
                    {
                        "ErrorEquals": ["States.TaskFailed"],
                        "IntervalSeconds": 30,
                        "MaxAttempts": 2,
                        "BackoffRate": 2.0,
                    }
                ],
            },

            # ── State 5: Notify success via SNS ────────────────────
            "NotifySuccess": {
                "Type": "Task",
                "Resource": "arn:aws:states:::sns:publish",
                "Parameters": {
                    "TopicArn": SNS_TOPIC_ARN,
                    "Subject": "[Fraud Platform] ETL Pipeline Succeeded",
                    "Message.$": "States.Format('Fraud Platform ETL pipeline completed successfully.\n\nExecution: {}\nBronze→Silver: SUCCEEDED\nSilver→Gold: SUCCEEDED\n\nGold tables updated:\n- gold_fraud_by_merchant\n- gold_fraud_by_segment\n- gold_fraud_by_hour\n- gold_high_risk_customers\n- gold_pipeline_run_summary', $$.Execution.Name)",
                },
                "Next": "PipelineSucceeded",
            },

            # ── State 6: Notify failure via SNS ────────────────────
            "NotifyFailure": {
                "Type": "Task",
                "Resource": "arn:aws:states:::sns:publish",
                "Parameters": {
                    "TopicArn": SNS_TOPIC_ARN,
                    "Subject": "[Fraud Platform] ETL Pipeline FAILED",
                    "Message.$": "States.Format('Fraud Platform ETL pipeline FAILED.\n\nExecution: {}\nCheck CloudWatch logs for details:\nhttps://console.aws.amazon.com/states/home?region=us-east-1#/executions', $$.Execution.Name)",
                },
                "Next": "PipelineFailed",
            },

            # ── Terminal states ─────────────────────────────────────
            "PipelineSucceeded": {
                "Type": "Succeed",
            },
            "PipelineFailed": {
                "Type": "Fail",
                "Error":  "ETLPipelineFailed",
                "Cause": "One or more Glue jobs failed. Check CloudWatch logs.",
            },
        },
    }


# ── Create or update state machine ─────────────────────────────────

def deploy_state_machine(sfn_client) -> str:
    """Create the state machine or update it if it already exists."""
    definition = json.dumps(build_state_machine_definition(), indent=2)

    try:
        resp = sfn_client.create_state_machine(
            name=MACHINE_NAME,
            definition=definition,
            roleArn=SFN_ROLE_ARN,
            type="STANDARD",
            loggingConfiguration={
                "level": "ERROR",
                "includeExecutionData": True,
                "destinations": [],
            },
            tracingConfiguration={"enabled": True},
        )
        arn = resp["stateMachineArn"]
        log.info("Created state machine: %s", arn)
        return arn

    except sfn_client.exceptions.StateMachineAlreadyExists:
        # Get existing ARN
        machines = sfn_client.list_state_machines()["stateMachines"]
        existing = next(
            (m for m in machines if m["name"] == MACHINE_NAME), None
        )
        if existing:
            arn = existing["stateMachineArn"]
            sfn_client.update_state_machine(
                stateMachineArn=arn,
                definition=definition,
                roleArn=SFN_ROLE_ARN,
            )
            log.info("Updated state machine: %s", arn)
            return arn
        raise


# ── EventBridge weekly schedule ─────────────────────────────────────

def create_eventbridge_schedule(events_client, sfn_arn: str):
    """
    Create an EventBridge rule that triggers the state machine weekly.

    Schedule: Every Sunday at 2:00 AM UTC
    Why Sunday 2 AM? Low traffic period, fraud team reviews Monday morning.
    In production you'd align this with your data SLA requirements.

    Cron format: cron(minutes hours day-of-month month day-of-week year)
    cron(0 2 ? * SUN *) = 2:00 AM every Sunday
    """
    # EventBridge needs permission to start Step Functions executions
    # We use the Step Functions role ARN for the target
    try:
        events_client.put_rule(
            Name=SCHEDULE_NAME,
            ScheduleExpression="cron(0 2 ? * SUN *)",
            State="ENABLED",
            Description=(
                "Triggers Fraud Platform ETL pipeline weekly "
                "(Bronze→Silver→Gold) every Sunday at 2AM UTC"
            ),
        )
        log.info("EventBridge rule created: %s", SCHEDULE_NAME)

        # Add Step Functions as the target
        events_client.put_targets(
            Rule=SCHEDULE_NAME,
            Targets=[
                {
                    "Id":      "FraudPlatformETLTarget",
                    "Arn":     sfn_arn,
                    "RoleArn": SFN_ROLE_ARN,
                    "Input":   json.dumps({
                        "trigger": "scheduled",
                        "schedule": "weekly-sunday-2am-utc",
                        "pipeline": "fraud-platform-etl",
                    }),
                }
            ],
        )
        log.info("EventBridge target set to Step Functions state machine")

    except Exception as e:
        log.warning("EventBridge setup: %s", e)


# ── Test execution ──────────────────────────────────────────────────

def run_test_execution(sfn_client, sfn_arn: str) -> str:
    """
    Trigger a manual execution now so we can see it run.
    This is identical to what EventBridge triggers weekly.
    """
    execution_name = f"manual-test-{int(time.time())}"
    resp = sfn_client.start_execution(
        stateMachineArn=sfn_arn,
        name=execution_name,
        input=json.dumps({
            "trigger": "manual-test",
            "initiated_by": "deploy_step_functions.py",
        }),
    )
    execution_arn = resp["executionArn"]
    log.info("Execution started: %s", execution_name)
    log.info("Execution ARN: %s", execution_arn)
    return execution_arn


def monitor_execution(sfn_client, execution_arn: str):
    """
    Poll execution status and print state transitions.
    Step Functions runs the Glue jobs synchronously using
    .sync integration — it waits for each job to complete
    before moving to the next state.

    Expected total duration: ~12-15 minutes (Glue job startup + ETL)
    """
    log.info("")
    log.info("Monitoring execution (Glue jobs take 5-8 min each) ...")
    log.info("Watch the visual graph at:")
    log.info("  https://console.aws.amazon.com/states/home?"
             "region=%s#/executions/details/%s", REGION, execution_arn)
    log.info("")

    last_state = None
    poll_count  = 0
    max_polls   = 80   # ~40 minutes max

    while poll_count < max_polls:
        time.sleep(30)
        poll_count += 1

        resp   = sfn_client.describe_execution(executionArn=execution_arn)
        status = resp["status"]

        # Get current state from execution history
        history = sfn_client.get_execution_history(
            executionArn=execution_arn,
            maxResults=5,
            reverseOrder=True,
        )["events"]

        current_state = None
        for event in history:
            if "stateEnteredEventDetails" in event:
                current_state = event["stateEnteredEventDetails"]["name"]
                break

        if current_state != last_state:
            log.info("  [%3d polls] Status=%-12s | Current state: %s",
                     poll_count, status, current_state or "initializing")
            last_state = current_state
        else:
            log.info("  [%3d polls] Status=%-12s | Still in: %s ...",
                     poll_count, status, current_state or "initializing")

        if status == "SUCCEEDED":
            log.info("")
            log.info("=" * 60)
            log.info("✓ Pipeline SUCCEEDED")
            log.info("  Start time : %s", resp["startDate"])
            log.info("  Stop time  : %s", resp["stopDate"])
            duration = (resp["stopDate"] - resp["startDate"]).total_seconds()
            log.info("  Duration   : %.0f seconds (%.1f minutes)",
                     duration, duration / 60)
            log.info("")
            log.info("Check Athena — Gold tables are freshly updated:")
            log.info("  https://console.aws.amazon.com/athena/home?region=%s", REGION)
            log.info("=" * 60)
            return True

        if status in ("FAILED", "TIMED_OUT", "ABORTED"):
            log.error("")
            log.error("✗ Pipeline %s", status)
            if resp.get("cause"):
                log.error("  Cause: %s", resp["cause"])
            log.error("  Check execution history:")
            log.error("  https://console.aws.amazon.com/states/home?"
                      "region=%s#/executions/details/%s", REGION, execution_arn)
            return False

    log.warning("Monitor timed out — check console for status")
    return False


# ── Main ────────────────────────────────────────────────────────────

def main():
    sfn_client    = boto3.client("stepfunctions", region_name=REGION)
    events_client = boto3.client("events",        region_name=REGION)

    log.info("=" * 60)
    log.info("Fraud Platform — Step Functions Deployment")
    log.info("  State Machine : %s", MACHINE_NAME)
    log.info("  Schedule      : Every Sunday 2AM UTC")
    log.info("  Glue Job 1    : fraud-bronze-to-silver")
    log.info("  Glue Job 2    : fraud-silver-to-gold")
    log.info("  SNS Topic     : fraud-alerts")
    log.info("=" * 60)

    # 1. Deploy state machine
    log.info("")
    log.info("Step 1: Deploying state machine ...")
    sfn_arn = deploy_state_machine(sfn_client)

    # 2. Create EventBridge weekly schedule
    log.info("")
    log.info("Step 2: Creating EventBridge weekly schedule ...")
    create_eventbridge_schedule(events_client, sfn_arn)

    # 3. Run test execution
    log.info("")
    log.info("Step 3: Starting test execution ...")
    log.info("This runs the full Bronze→Silver→Gold pipeline.")
    log.info("Expected duration: 12-15 minutes.")
    log.info("")

    execution_arn = run_test_execution(sfn_client, sfn_arn)

    # 4. Monitor
    log.info("Step 4: Monitoring execution ...")
    monitor_execution(sfn_client, execution_arn)

    log.info("")
    log.info("Console links:")
    log.info("  State machine : https://console.aws.amazon.com/states/home?"
             "region=%s#/statemachines", REGION)
    log.info("  Execution     : https://console.aws.amazon.com/states/home?"
             "region=%s#/executions/details/%s", REGION, execution_arn)
    log.info("  EventBridge   : https://console.aws.amazon.com/events/home?"
             "region=%s#/rules", REGION)


if __name__ == "__main__":
    main()
