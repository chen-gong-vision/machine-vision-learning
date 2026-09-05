# -*- coding: utf-8 -*-
"""
MySQL 连通验证 + 最小 CRUD 模板（陈工 M2 之前的预热）

前置：
    1) 双击 G:\MySQL\init_mysql.bat 初始化（仅需一次）
    2) 双击 G:\MySQL\start_mysql.bat 启动（保持窗口开着）
    3) venv 已装 mysql-connector-python（pip install mysql-connector-python）

运行：在 venv 里  python mysql_hello.py
预期：建库 demo -> 建表 t1 -> 插一条 -> 查出来打印 -> 删表（干净退出）

⚠️ 若报错 "Can't connect"：99% 是 MySQL 没启动，去开 start_mysql.bat。
⚠️ 驱动提示：本模板用 mysql-connector-python（教程标准）。若该 import/connect 在某些环境段错误，
   改 pip install pymysql 并把 mysql.connector.connect(...) 换成 pymysql.connect(...) 即可（接口一致）。
"""
import mysql.connector
from mysql.connector import errorcode

HOST, PORT, USER, PASSWORD = "127.0.0.1", 3306, "root", ""


def main():
    try:
        conn = mysql.connector.connect(
            host=HOST, port=PORT, user=USER, password=PASSWORD
        )
    except mysql.connector.Error as e:
        if e.errno == errorcode.ER_ACCESS_DENIED_ERROR:
            print("✗ 账号/密码错（当前 root 密码为空，检查 PASSWORD 变量）")
        else:
            print(f"✗ 连不上 MySQL：{e}\n   → 先双击 G:\\MySQL\\start_mysql.bat 启动服务端")
        return

    print(f"✓ 连上 MySQL，server 版本：{conn.get_server_info()}")
    cur = conn.cursor()

    cur.execute("CREATE DATABASE IF NOT EXISTS demo")
    cur.execute("USE demo")
    cur.execute(
        "CREATE TABLE IF NOT EXISTS t1 ("
        "id INT PRIMARY KEY AUTO_INCREMENT, name VARCHAR(50), ts DATETIME DEFAULT CURRENT_TIMESTAMP)"
    )
    cur.execute("INSERT INTO t1(name) VALUES (%s)", ("hello_mysql",))
    conn.commit()

    cur.execute("SELECT id, name, ts FROM t1 ORDER BY id DESC LIMIT 5")
    rows = cur.fetchall()
    print(f"✓ 查到 {len(rows)} 条记录：")
    for r in rows:
        print("   ", r)

    # 实验完清理，保持库干净
    cur.execute("DROP TABLE IF EXISTS t1")
    conn.commit()
    print("✓ 已清理临时表 t1，连通验证通过 ✅")

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
