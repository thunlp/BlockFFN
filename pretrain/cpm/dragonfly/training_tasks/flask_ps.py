import logging
from multiprocessing import Lock

from flask import Flask
from flask import jsonify
from flask import request

app = Flask(__name__)

# 获取 Werkzeug 日志记录器并设置日志级别
log = logging.getLogger("werkzeug")
log.setLevel(logging.WARNING)


class GlobalAvgTokensStat(object):
    def __init__(self, decay_factor: float = 0.98):
        self._avg_tokens = {}
        self.decay_factor = decay_factor
        self.lock = Lock()
        self.task_locks = {}

    def set_avg_tokens(self, task_name, avg_tokens):
        self._register_task_lock_helper(task_name)
        with self.task_locks[task_name]:
            self._avg_tokens[task_name] = avg_tokens

    def update_avg_tokens_by_ema(self, task_name, length):
        self._register_task_lock_helper(task_name)
        with self.task_locks[task_name]:
            if task_name in self._avg_tokens and self._avg_tokens[task_name] > 0:
                self._avg_tokens[task_name] = self._avg_tokens[task_name] * self.decay_factor + length * (
                    1 - self.decay_factor
                )
            else:
                self._avg_tokens[task_name] = length

    def get_avg_tokens(self, task_name):
        self._register_task_lock_helper(task_name)
        with self.task_locks[task_name]:
            return self._avg_tokens.get(task_name, -1)

    def _register_task_lock_helper(self, task_name):
        if task_name not in self.task_locks:
            with self.lock:
                if task_name not in self.task_locks:
                    self.task_locks[task_name] = Lock()


global_avg_tokens_stat = GlobalAvgTokensStat()


@app.route("/avg_tokens/<path:task_name>", methods=["GET"])
def get_avg_tokens(task_name):
    global global_avg_tokens_stat
    avg_tokens = global_avg_tokens_stat.get_avg_tokens(task_name)
    return jsonify({"avg_tokens": avg_tokens})


@app.route("/avg_tokens/<path:task_name>", methods=["POST"])
def set_avg_tokens(task_name):
    global global_avg_tokens_stat
    action = request.args.get("action", "update", type=str)
    length = request.args.get("length", -1, type=int)
    if action == "set":
        global_avg_tokens_stat.set_avg_tokens(task_name, length)
    elif action == "update":
        global_avg_tokens_stat.update_avg_tokens_by_ema(task_name, length)
    else:
        raise ValueError(f"Unknown action: {action}")
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(port=5000, debug=True)
