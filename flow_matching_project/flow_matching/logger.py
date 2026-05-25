import csv
import datetime
import os
import tempfile
from collections import defaultdict


class _Logger:
    def __init__(self, log_dir):
        self.dir = log_dir
        self.log_path = os.path.join(log_dir, "log.txt")
        self.csv_path = os.path.join(log_dir, "progress.csv")
        self.name2val = defaultdict(float)
        self.name2cnt = defaultdict(int)
        self.csv_keys = []

    def log(self, *args):
        text = " ".join(str(x) for x in args)
        print(text, flush=True)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(text + "\n")

    def logkv(self, key, val):
        self.name2val[key] = val

    def logkv_mean(self, key, val):
        oldval, cnt = self.name2val[key], self.name2cnt[key]
        self.name2val[key] = oldval * cnt / (cnt + 1) + val / (cnt + 1)
        self.name2cnt[key] = cnt + 1

    def dumpkvs(self):
        row = dict(self.name2val)
        if not row:
            return row

        new_keys = [k for k in row.keys() if k not in self.csv_keys]
        if new_keys:
            self.csv_keys.extend(sorted(new_keys))
            existing_rows = []
            if os.path.exists(self.csv_path):
                with open(self.csv_path, "r", newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    existing_rows = list(reader)
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.csv_keys)
                writer.writeheader()
                for existing_row in existing_rows:
                    writer.writerow(existing_row)

        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.csv_keys)
            writer.writerow({k: row.get(k) for k in self.csv_keys})

        display = " | ".join(f"{k}: {row[k]}" for k in sorted(row.keys()))
        self.log(display)
        self.name2val.clear()
        self.name2cnt.clear()
        return row


_CURRENT = None


def configure(dir=None):
    global _CURRENT
    if dir is None:
        dir = os.getenv("OPENAI_LOGDIR")
    if dir is None:
        dir = os.path.join(
            tempfile.gettempdir(),
            datetime.datetime.now().strftime("fm-%Y-%m-%d-%H-%M-%S-%f"),
        )
    os.makedirs(dir, exist_ok=True)
    _CURRENT = _Logger(dir)
    _CURRENT.log(f"Logging to {dir}")


def _get():
    global _CURRENT
    if _CURRENT is None:
        configure()
    return _CURRENT


def log(*args):
    _get().log(*args)


def logkv(key, val):
    _get().logkv(key, val)


def logkv_mean(key, val):
    _get().logkv_mean(key, val)


def dumpkvs():
    return _get().dumpkvs()


def get_dir():
    return _get().dir
