import json, logging, sys, time

class JsonFormatter(logging.Formatter):
    def format(self, record):
        data = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "level": record.levelname,
            "msg": record.getMessage(),
            "logger": record.name,
        }
        if record.exc_info:
            data["exc_info"] = self.formatException(record.exc_info)
        if hasattr(record, "context"):
            data.update(record.context)
        return json.dumps(data, ensure_ascii=False)

def setup_logging(level="INFO"):
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = []
    root.addHandler(h)
    root.setLevel(level)