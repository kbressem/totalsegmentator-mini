import logging
import time
from typing import List, Optional

import ignite
import requests
import yaml


class PushnotificationHandler:
    logger = logging.getLogger(__name__)

    def __init__(self, credentials: Optional[str] = None, identifier: str = "Study") -> None:
        """Send push notifications with pushover for remote monitoring
        Args:
            credentials: YAML file containing the `app_token`,
                `user_key` and, if you are behind a proxy `proxies`.
                For more information on pushover visit: https://support.pushover.net/
            identifier: Printed at the beginning of each message. 
        """
        if not credentials: 
            self.logger.warning(
                "No pushover credentials file submitted, "
                "will not try to push trainings progress to pushover device. "
                "If you want to receive status updated via pushover, provide the "
                "path to a yaml file, containing the `app_token`, `user_key` and `proxies`."
            )
            self.enable_notifications = False
        else:
            with open(credentials, "r") as stream:
                credentials = yaml.safe_load(stream)
            self.app_token = credentials["app_token"]
            self.user_key = credentials["user_key"]
            self.proxies = credentials["proxies"] if "proxies" in credentials else None
            self.enable_notifications = True

        self.identifier = identifier 
        self.key_metric = -1
        self.improvement = False

    def attach(self, engine: ignite.engine.Engine) -> None:
        """
        Args:
            engine: Ignite Engine, should be an evaluator with metrics.
        """
        if self.enable_notifications:
            engine.add_event_handler(ignite.engine.Events.STARTED, self.start_training)
            engine.add_event_handler(ignite.engine.Events.COMPLETED, self.push_metrics)
            engine.add_event_handler(ignite.engine.Events.TERMINATE, self.push_terminated)

    def push(self, message: str, priority: int = -1):
        "Send message to device"
        _ = requests.post(
            "https://api.pushover.net/1/messages.json",
            data={
                "token": self.app_token,
                "user": self.user_key,
                "message": message,
                "priority": priority,
                "html": 1,  # enable html formatting
            },
            proxies=self.proxies,
        )

    def _get_metrics(self, engine: ignite.engine.Engine) -> str:
        "Extract metrics from engine.state"
        message = ""
        metric_names = list(engine.state.metrics.keys())

        key_metric = engine.state.metrics[metric_names[0]]
        self.improvement = key_metric > self.key_metric
        self.key_metric = max(key_metric, self.key_metric)

        for mn in metric_names:
            message += f"{mn}: {engine.state.metrics[mn]:.5f}\n"
        return message

    def start_training(self, engine: ignite.engine.Engine) -> None:
        "Collect basic data for run"
        self.start_time = time.time()

    def push_metrics(self, engine: ignite.engine.Engine) -> None:
        epoch = engine.state.epoch
        message = f"<b>{self.identifier}:</b>\n"
        message += f"Metrics after epoch {epoch}:\n"
        message += self._get_metrics(engine)
        if self.improvement:
            self.push(message)

    def push_terminated(self, engine: ignite.engine.Engine) -> None:
        end_time = time.time()
        seconds = self.start_time - end_time
        minutes = seconds // 60
        hours = minutes // 60
        minutes = minutes % 60
        duration = f"{hours}:{minutes}:{seconds}"
        epoch = engine.state.epoch
        message = f"<b>{self.identifier}:</b>\n"
        message += f"Training ended after {epoch} epochs\n"
        message += f"Duration {duration} \n"
        message += self._get_metrics(engine)
        self.push(message)

    def push_exception(self, engine: ignite.engine.Engine) -> None:
        epoch = engine.state.epoch
        message = f"<b>{self.identifier}:</b>\n"
        message += f"Exception raise after {epoch}\n"
        self.push(message, 0)