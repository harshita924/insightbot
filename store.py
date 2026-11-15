"""Audit trail: every question, SQL, answer, status, user feedback and analyst resolution (SQLite)."""
import os, sqlite3, datetime as dt, json

DB = os.environ.get("INSIGHTBOT_LOG_DB", os.path.join(os.path.dirname(os.path.abspath(__file__)), "insightbot_log.db"))

def _c(path=None):
    c = sqlite3.connect(path or DB, check_same_thread=False)
    c.execute("""CREATE TABLE IF NOT EXISTS log(
        id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, question TEXT, sub_question TEXT, status TEXT, label TEXT,
        mode TEXT, sql TEXT, answer TEXT, latency_ms INTEGER, feedback TEXT DEFAULT '', flag_note TEXT DEFAULT '',
        review_status TEXT DEFAULT '', analyst_note TEXT DEFAULT '')""")
    return c

def log_result(res, path=None) -> list[int]:
    """Write one row per part; returns the row ids."""
    c = _c(path); ids = []
    for p in res["parts"]:
        needs_review = "open" if p["status"] in ("escalated", "blocked") else ""
        cur = c.execute("INSERT INTO log(ts,question,sub_question,status,label,mode,sql,answer,latency_ms,review_status) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (dt.datetime.now().isoformat(timespec="seconds"), res["question"], p["sub_question"], p["status"], p.get("label", ""),
             p.get("mode", ""), p.get("sql", ""), p.get("narrative") or p.get("message", ""), res["latency_ms"], needs_review))
        ids.append(cur.lastrowid)
    c.commit(); c.close(); return ids

def set_feedback(row_id, feedback, note="", path=None):
    c = _c(path)
    if feedback == "flagged":
        c.execute("UPDATE log SET feedback=?, flag_note=?, review_status='open' WHERE id=?", (feedback, note, row_id))
    else:
        c.execute("UPDATE log SET feedback=? WHERE id=?", (feedback, row_id))
    c.commit(); c.close()

def resolve(row_id, note, path=None):
    c = _c(path); c.execute("UPDATE log SET review_status='resolved', analyst_note=? WHERE id=?", (note, row_id)); c.commit(); c.close()

def read(path=None):
    import pandas as pd
    c = _c(path); df = pd.read_sql_query("SELECT * FROM log ORDER BY id DESC", c); c.close(); return df
