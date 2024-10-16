import time
import logging
from locust import HttpUser, task, between, events
from locust.runners import STATE_STOPPED

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Переменные для гибкого изменения параметров нагрузки
# Максимальное количество пользователей
MAX_USERS = 100
# Шаг увеличения пользователей
STEP_USERS = 10
# Порог RPS для остановки
RPS_THRESHOLD = 50
# Максимальная допустимая задержка (в секундах)
LATENCY_THRESHOLD = 1.0
# Максимальное число ошибок для остановки
ERROR_THRESHOLD = 5


class LoadTestUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def test_health_check(self):
        self.client.get("/health")

    @task
    def test_metrics(self):
        self.client.get("/metrics")


errors_count = 0


@events.request_failure.add_listener
def log_failure(request_type, name, response_time, response_length, exception, **kwargs):
    global errors_count
    errors_count += 1
    logger.error(f"Request failed: {name}, {exception}")
    if errors_count >= ERROR_THRESHOLD:
        logger.error(f"Error threshold exceeded: {errors_count} errors. Stopping test.")
        stop_test()


@events.request_success.add_listener
def log_success(request_type, name, response_time, response_length, **kwargs):
    if response_time > LATENCY_THRESHOLD * 1000:
        logger.warning(f"High latency: {response_time}ms for request {name}")
        if response_time / 1000 > LATENCY_THRESHOLD:
            logger.error(f"Latency threshold exceeded: {response_time / 1000:.2f}s. Stopping test.")
            stop_test()


def stop_test():
    logger.info("Stopping test due to issues...")
    for runner in events.environment.runner.all_runners:
        runner.quit()
    if events.environment.runner:
        events.environment.runner.state = STATE_STOPPED


@events.test_stop.add_listener
def on_test_stop(**kwargs):
    logger.info(f"Test finished with {errors_count} errors.")
