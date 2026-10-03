import time
import json
import smtplib
import threading
import queue
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from app.config import load_system_config
from app.database import (
    db_queue_alert, db_get_pending_alerts,
    db_mark_alert_dispatched, db_increment_alert_retry
)

try:
    import paho.mqtt.client as mqtt
except ImportError:
    mqtt = None


class AlertDispatcher:
    def __init__(self):
        self.in_memory_queue = queue.Queue(maxsize=1000)
        self.running = True
        self.mqtt_client = None

        self._init_mqtt()
        self.worker_thread = threading.Thread(target=self._dispatcher_worker, daemon=True)
        self.worker_thread.start()

    def _init_mqtt(self):
        cfg = load_system_config().get("alerts", {}).get("channels", {}).get("mqtt", {})
        if mqtt is not None and cfg.get("enabled", False):
            try:
                self.mqtt_client = mqtt.Client(client_id=cfg.get("client_id", "IBVAP_NODE"))
                self.mqtt_client.connect(
                    cfg.get("broker_host", "localhost"),
                    cfg.get("broker_port", 1883),
                    keepalive=cfg.get("keepalive", 60)
                )
                self.mqtt_client.loop_start()
                print("[ALERTS] MQTT client connected to defense broker.")
            except Exception as e:
                print(f"[ALERTS] MQTT connection failed: {e}. Operating in store-and-forward mode.")
                self.mqtt_client = None

    def dispatch(self, event_data: dict):
        payload_str = json.dumps(event_data)
        try:
            db_queue_alert(
                event_id=event_data.get("id", int(time.time() * 1000)),
                threat_type=event_data.get("type", "UNKNOWN"),
                level=event_data.get("level", "CODE RED"),
                payload_json=payload_str
            )
        except Exception as e:
            print(f"[ALERTS] Database queueing warning: {e}")

        try:
            self.in_memory_queue.put_nowait(event_data)
        except queue.Full:
            pass

    def _dispatcher_worker(self):
        while self.running:
            try:
                event = self.in_memory_queue.get(timeout=2.0)
                self._send_to_all_channels(event)
                self.in_memory_queue.task_done()
            except queue.Empty:
                pass

            self._retry_pending_backlog()

    def _send_to_all_channels(self, event: dict):
        cfg = load_system_config().get("alerts", {}).get("channels", {})

        if cfg.get("webhook", {}).get("enabled", False):
            self._send_webhook(cfg["webhook"], event)

        if cfg.get("mqtt", {}).get("enabled", False):
            self._send_mqtt(cfg["mqtt"], event)

        if cfg.get("email_smtp", {}).get("enabled", False):
            self._send_email(cfg["email_smtp"], event)

        if cfg.get("sms_webhook", {}).get("enabled", False):
            self._send_sms(cfg["sms_webhook"], event)

    def _send_webhook(self, conf: dict, event: dict):
        url = conf.get("url")
        if not url:
            return
        try:
            headers = conf.get("headers", {"Content-Type": "application/json"})
            timeout = conf.get("timeout_sec", 3.0)
            res = requests.post(url, json=event, headers=headers, timeout=timeout)
            if res.status_code < 300:
                db_mark_alert_dispatched(event.get("id"))
            else:
                db_increment_alert_retry(event.get("id"))
        except Exception:
            db_increment_alert_retry(event.get("id"))

    def _send_mqtt(self, conf: dict, event: dict):
        if self.mqtt_client is not None:
            try:
                topic = conf.get("topic", "defense/bop/perimeter_alerts")
                self.mqtt_client.publish(topic, json.dumps(event), qos=1)
            except Exception:
                pass

    def _send_email(self, conf: dict, event: dict):
        try:
            msg = MIMEMultipart()
            msg['From'] = conf.get("sender_email")
            msg['To'] = ", ".join(conf.get("recipient_emails", []))
            msg['Subject'] = f"🚨 IBVAP DEFENSE ALERT: {event.get('level')} - {event.get('type')}"

            body = (
                f"TACTICAL PERIMETER BREACH REPORT\n"
                f"----------------------------------------\n"
                f"Timestamp:   {event.get('time')}\n"
                f"Sensor ID:   {event.get('camera_id')}\n"
                f"Threat Code: {event.get('level')}\n"
                f"Event Type:  {event.get('type')}\n"
                f"Targets:     {event.get('targets')}\n"
                f"Status:      {event.get('status')}\n"
            )
            msg.attach(MIMEText(body, 'plain'))

            server = smtplib.SMTP(conf.get("smtp_host"), conf.get("smtp_port"), timeout=5)
            if conf.get("use_tls", True):
                server.starttls()
            server.send_message(msg)
            server.quit()
        except Exception as e:
            print(f"[ALERTS] SMTP transmission failed: {e}")

    def _send_sms(self, conf: dict, event: dict):
        url = conf.get("url")
        phones = conf.get("phone_numbers", [])
        valid_phones = [p for p in phones if p and not p.startswith("+91XXXX")]
        if not url or not valid_phones:
            return
        try:
            payload = {
                "numbers": valid_phones,
                "message": f"IBVAP {event.get('level')}: {event.get('type')} at {event.get('camera_id')} [{event.get('time')}]"
            }
            requests.post(url, json=payload, timeout=4.0)
        except Exception:
            pass

    def _retry_pending_backlog(self):
        saf_cfg = load_system_config().get("alerts", {}).get("store_and_forward", {})
        if not saf_cfg.get("enabled", True):
            return

        pending = db_get_pending_alerts(max_retries=saf_cfg.get("max_retries", 5))
        for row in pending:
            evt_id, payload_json, retries = row
            try:
                event = json.loads(payload_json)
                wh_cfg = load_system_config().get("alerts", {}).get("channels", {}).get("webhook", {})
                if wh_cfg.get("enabled", False):
                    res = requests.post(wh_cfg["url"], json=event, timeout=3.0)
                    if res.status_code < 300:
                        db_mark_alert_dispatched(evt_id)
                    else:
                        db_increment_alert_retry(evt_id)
                else:
                    db_mark_alert_dispatched(evt_id)
            except Exception:
                db_increment_alert_retry(evt_id)


alert_dispatcher = AlertDispatcher()