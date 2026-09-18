'''
数据库初始化脚本设计：
1.先准备数据路径
2.读取原始订单数据
3.连接数据库
4.创建数据表
5.导入数据
6.处理重复数据
7.返回结果
'''
import sqlite3
import json
from pathlib import Path

from config import setting


# 数据文件、数据库文件默认与当前脚本存放在同一目录下。
# 使用脚本自身路径定位文件，可避免程序从不同工作目录运行时找不到数据。



def setup_database(
    json_path: str | Path = setting.JSON_PATH,
    database_path: str | Path = setting.JSON_PATH,
) -> int:
    """创建订单表，并将 JSON 文件中的订单导入 SQLite 数据库。

    如果数据库中已经存在相同 order_id 的订单，则更新该订单的数据，
    因此可以重复执行此函数而不会产生重复订单。

    返回成功处理的订单数量。
    """
    # 将传入的字符串路径统一转换为 Path 对象，便于后续文件操作。
    json_path = Path(json_path)
    database_path = Path(database_path)

    # orders.json 使用 UTF-8 编码，确保中文商品名称和订单状态正常读取。
    with json_path.open("r", encoding="utf-8") as file:
        orders = json.load(file)

    # JSON 文件的顶层必须是订单数组，避免传入错误格式的数据。
    if not isinstance(orders, list):
        raise ValueError("orders.json 的顶层结构必须是数组")

    # 如果数据库的父目录不存在，则自动创建。
    database_path.parent.mkdir(parents=True, exist_ok=True)

    # 使用 with 管理数据库连接：执行成功时自动提交，发生异常时自动回滚。
    with sqlite3.connect(database_path) as connection:
        # 创建订单表。IF NOT EXISTS 可保证重复运行时不会因表已存在而报错。
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS orders (
                -- 订单编号唯一，用作主键
                order_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                product_name TEXT NOT NULL,
                -- 商品数量必须大于 0
                quantity INTEGER NOT NULL CHECK (quantity > 0),
                -- 订单金额不能为负数
                total_price REAL NOT NULL CHECK (total_price >= 0),
                order_status TEXT NOT NULL,
                -- 物流单号和预计送达日期允许为空
                tracking_number TEXT,
                estimated_delivery TEXT,
                order_date TEXT NOT NULL
            )
            """
        )

        # 使用命名占位符批量写入订单，避免手动拼接 SQL 带来的错误和风险。
        connection.executemany(
            """
            INSERT INTO orders (
                order_id,
                user_id,
                product_name,
                quantity,
                total_price,
                order_status,
                tracking_number,
                estimated_delivery,
                order_date
            ) VALUES (
                :order_id,
                :user_id,
                :product_name,
                :quantity,
                :total_price,
                :order_status,
                :tracking_number,
                :estimated_delivery,
                :order_date
            )
            -- 如果 order_id 已存在，则使用 JSON 中的最新内容更新原记录。
            ON CONFLICT(order_id) DO UPDATE SET
                user_id = excluded.user_id,
                product_name = excluded.product_name,
                quantity = excluded.quantity,
                total_price = excluded.total_price,
                order_status = excluded.order_status,
                tracking_number = excluded.tracking_number,
                estimated_delivery = excluded.estimated_delivery,
                order_date = excluded.order_date
            """,
            orders,
        )

    # 返回本次从 JSON 文件中读取并处理的订单总数。
    return len(orders)


if __name__ == "__main__":
    # 直接运行本脚本时执行数据库初始化，并输出导入结果。
    imported_count = setup_database()
    print(f"成功导入 {imported_count} 条订单到 {setting.JSON_PATH}")
