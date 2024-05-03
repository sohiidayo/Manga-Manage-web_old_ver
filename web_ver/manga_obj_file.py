#处理漫画文件相关的文档

#本人的漫画都是压缩包的形式(zip，7z)的存在，部分是同一本漫画有不同汉化组的版本，也有连载漫画不同期的部分
#我并不想给不同漫画中的一样的页而增加存储成本，所以定义这个模块，用于导入漫画，管理漫画
#在本模块内这些漫画是以 漫画名-索引列，页-漫画 的的形式存在的
#在模块外，这些漫画是以一本存在的，通过漫画标题获取整个漫画的文件

import re
import time

last_time = time.time()
def TestTime():
    global last_time
    current_time = time.time()
    elapsed_time = current_time - last_time
    last_time = current_time
    print(f"Elapsed Time: {elapsed_time} seconds")
#TestTime()


import datetime
import hashlib
import json
from PIL import Image
import numpy as np
import io
import threading
import os
import sqlite3
import concurrent.futures
import imagehash
import py7zr
import zipfile
from concurrent.futures import ThreadPoolExecutor
import psutil
import os
from PIL import Image
from io import BytesIO
import numpy as np
from pywebio.output import put_grid, put_image

from pywebio.platform.flask import start_server
from pywebio import STATIC_PATH
from pywebio.output import *
from pywebio.input import *
from pywebio import start_server
from pywebio.session import  *
from pywebio.pin import  *

#计算图片的md5，可以是路径、字节流、Numpy数组、PIL对象
def CalculateImgMd5(img):
    """
    计算图片的 MD5 哈希值。

    参数：
    - img：图片数据，可以是路径字符串、字节流、NumPy 数组、PIL 图像对象。

    返回值：
    - md5_hash：图片的 MD5 哈希值，字符串格式。

    异常：
    - 如果图片格式不支持，则抛出 ValueError 异常。
    """
    # 将不同格式的图像统一转换为 PIL 图像对象
    if isinstance(img, str):  # 如果是路径字符串
        img = Image.open(img)
    elif isinstance(img, bytes):  # 如果是字节流
        img = Image.open(io.BytesIO(img))
    elif isinstance(img, np.ndarray):  # 如果是 NumPy 数组
        img = Image.fromarray(img)
    elif img is None:  # 如果是空
        return None
    elif not isinstance(img, Image.Image):  # 如果不是 PIL 图像对象
        raise ValueError("Unsupported image format")

    # 计算图像的 MD5 值
    md5_hash = hashlib.md5(img.tobytes()).hexdigest()
    return md5_hash

# # 示例用法
# img_path = "/path/to/image.jpg"
# img_bytes = b'\x89\x50\x4e\x47\x0d\x0a\x1a\x0a\x00\x00\x00\x0d\x49\x48\x44\x52'  # 示例字节流
# img_numpy = np.zeros((10, 10, 3), dtype=np.uint8)  # 示例 NumPy 数组
# img_pil = Image.open("/path/to/image.jpg")  # 示例 PIL 图像对象
#
# md5_1 = calculate_md5(img_path)
# md5_2 = calculate_md5(img_bytes)
# md5_3 = calculate_md5(img_numpy)
# md5_4 = calculate_md5(img_pil)
#
# print(md5_1)
# print(md5_2)
# print(md5_3)
# print(md5_4)

#计算图片的phash，可以是路径、字节流、Numpy数组、PIL对象
def CalculateImgPhash(img):
    """
    计算图片的感知哈希（phash）。

    参数：
    - img：图片数据，可以是路径字符串、字节流、NumPy 数组、PIL 图像对象。

    返回值：
    - phash_str：图片的感知哈希值，字符串格式。

    异常：
    - 如果图片格式不支持，则抛出 ValueError 异常。
    """
    # 将不同格式的图像统一转换为 PIL 图像对象
    if isinstance(img, str):
        img = Image.open(img)
    elif isinstance(img, bytes):
        img = Image.open(io.BytesIO(img))
    elif isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    elif img is None:  # 如果是空
        return None
    elif not isinstance(img, Image.Image):
        raise ValueError("Unsupported image format")

    # 进行感知哈希处理并计算哈希值
    phash = imagehash.phash(img)
    phash_str = str(phash)
    return phash_str


def image_to_webp_bytes(img):
    """
    将图像转换为 WebP 格式的字节流输出。

    参数:
        img: 图像数据，可以是路径字符串、字节流、NumPy 数组或 PIL 图像对象。

    返回值:
        bytes: 转换后的图像数据，以字节流形式表示。

    异常:
        ValueError: 当图像格式不受支持时引发异常。
    """
    # 将不同格式的图像统一转换为 PIL 图像对象
    if isinstance(img, str):  # 如果是路径字符串
        img = Image.open(img)
    elif isinstance(img, bytes):  # 如果是字节流
        img = Image.open(io.BytesIO(img))
    elif isinstance(img, np.ndarray):  # 如果是 NumPy 数组
        img = Image.fromarray(img)
    elif img is None:  # 如果是空
        return None
    elif not isinstance(img, Image.Image):  # 如果不是 PIL 图像对象
        raise ValueError("Unsupported image format")

    # 创建一个 BytesIO 对象来保存图像的字节流
    img_byte_array = io.BytesIO()

    # 将图像保存为 WebP 格式到 BytesIO 对象
    img.save(img_byte_array, format='WEBP')

    # 获取字节流数据
    webp_bytes = img_byte_array.getvalue()

    return webp_bytes



def process_images_in_archive(archive_path, enable_multithreading=True):
    """
    对压缩包内部的图片进行 MD5 和感知哈希（phash）计算，并输出元组列表。
    输入路径元组列表，返回结果元组列表

    参数：
    - archive_path：压缩包路径（7z 或 zip 格式）。
    - enable_multithreading：是否启用多线程。默认为 True，表示启用多线程；False 表示不启用多线程。

    返回值：
    - image_info_list：包含图片信息的元组列表，每个元组格式为 (错误类型，MD5 哈希值，感知哈希值，图片字节流)。
      如果出现错误，错误类型为异常信息字符串，MD5 哈希值和感知哈希值为 None，图片字节流为 None。
    """
    image_info_list = []

    def process_image(entry):
        try:
            img_data = z.read(entry)
            img = Image.open(io.BytesIO(img_data))
            md5_hash = CalculateImgMd5(img)
            phash = CalculateImgPhash(img)
            return (None, md5_hash, phash, img_data)
        except Exception as e:
            return (str(e), None, None, None)

    try:
        if archive_path.endswith('.7z'):
            with py7zr.SevenZipFile(archive_path, mode='r') as z:
                entries = [entry for entry in z.getnames() if any(entry.endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp'])]

                if enable_multithreading:
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        results = executor.map(process_image, entries)
                else:
                    results = map(process_image, entries)

                image_info_list.extend(results)

        elif archive_path.endswith('.zip'):
            with zipfile.ZipFile(archive_path, mode='r') as z:
                entries = [entry for entry in z.namelist() if any(entry.endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp'])]

                if enable_multithreading:
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        results = executor.map(process_image, entries)
                else:
                    results = map(process_image, entries)

                image_info_list.extend(results)

    except Exception as e:
        image_info_list.append((str(e), None, None, None))

    return image_info_list

def save_image_as_webp(imgs, save_path=None, multi_thread=True):
    """
    将图像保存为 WebP 格式。

    参数：
    - imgs：输入的图像，可以是单个图像，也可以是元组列表。
    - save_path：保存路径的文件夹路径，默认为 None。如果为 None，则保存到当前目录下的 MangaImg 文件夹中。
    - multi_thread：是否使用多线程保存图像，默认为 True。

    返回值：
    - save_files：保存的文件路径列表。

    异常：
    - ValueError：如果输入的图像格式不支持。
    """
    save_files = []
    if isinstance(imgs, tuple):  # 如果是元组列表
        imgs = list(imgs)
    elif not isinstance(imgs, list):  # 如果是单个图像
        imgs = [imgs]

    if not multi_thread or len(imgs) == 1:  # 不使用多线程或只有一个图像
        for img in imgs:
            if  isinstance(img, tuple):
                img = img[0]
            save_file = _save_single_image_as_webp(img, save_path,save_files=[])
            save_files.append(save_file)
    else:  # 使用多线程保存图像
        threads = []
        for img in imgs:
            thread = threading.Thread(target=_save_single_image_as_webp, args=(img, save_path, save_files))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

    return save_files

def _save_single_image_as_webp(img, save_path, save_files=[]):
    """
    内部方法：将单个图像保存为 WebP 格式。

    参数：
    - img：输入的图像，可以是路径字符串、字节流、NumPy 数组、PIL 对象。
    - save_path：保存路径的文件夹路径，默认为 None。如果为 None，则保存到当前目录下的 MangaImg 文件夹中。
    - save_files：保存的文件路径列表。

    异常：
    - ValueError：如果输入的图像格式不支持。
    """
    # 将不同格式的图像统一转换为 PIL 图像对象,返回md5值
    if isinstance(img, str):  # 如果是路径字符串
        img = Image.open(img)
    elif isinstance(img, bytes):  # 如果是字节流
        img = Image.open(io.BytesIO(img))
    elif isinstance(img, np.ndarray):  # 如果是 NumPy 数组
        img = Image.fromarray(img)
    elif img is None:  # 如果是空
        return None
    elif not isinstance(img, Image.Image):  # 如果不是 PIL 图像对象
        raise ValueError("Unsupported image format")

    # 计算图像的 MD5 值
    md5_hash = CalculateImgMd5(img)

    # 获取保存文件夹路径
    if save_path is None:
        save_dir = "./MangaImg"
    else:
        save_dir = os.path.join(save_path, "MangaImg")

    # 如果文件夹不存在，则创建它
    os.makedirs(save_dir, exist_ok=True)

    # 拼接保存文件路径
    save_file = os.path.join(save_dir, md5_hash)

    # 添加 WebP 格式后缀
    save_file += ".webp"

    # 判断文件是否已存在
    if os.path.exists(save_file):
        return

    # 保存图像为 WebP 格式
    try:
        img.save(save_file, "WEBP")
    except Exception:
        pass
    # 将保存的文件路径添加到列表中
    save_files.append(save_file)

def generate_thumbnail(image, width=120):
    """
    生成指定大小的缩略图

    参数:
    - image: PIL图片对象，输入的原始图片
    - width: int，缩略图的宽度，默认为120像素

    返回值:
    - PIL图片对象，生成的缩略图

    示例用法:
    >>> image = Image.open('input.jpg')  # 打开图片文件
    >>> thumbnail = generate_thumbnail(image, width=200)  # 生成宽度为200像素的缩略图
    >>> thumbnail.show()  # 显示缩略图

    注意:
    - 如果原始图片的长宽比与指定宽度的长宽比不同，缩略图的高度会根据原始图片的长宽比进行调整，以保持比例不变。
    - 生成的缩略图是根据原始图片按比例缩放而来的，可能会有一定的失真。
    """
    # 获取原始图像的尺寸
    original_width, original_height = image.size

    # 计算缩略图的高度，保持长宽比不变
    height = int(width * original_height / original_width)

    # 生成缩略图
    thumbnail = image.resize((width, height))

    return thumbnail


import os
from PIL import Image
import numpy as np
import concurrent.futures

def read_image_from_md5(md5, path=None, type="pil", use_threading=True):
    """
    从指定路径文件夹或默认的当前目录下的"MangaImg"文件夹中读取图片。

    参数:
        md5 (str or list): 要读取的图片的MD5字符串，或MD5字符串列表。
        path (str, optional): 图片所在的路径。默认为None，表示从当前目录下的"MangaImg"文件夹中读取。
        type (str, optional): 图片读取类型，可以是'pil'或'numpy'。默认为'pil'。
        use_threading (bool, optional): 是否使用多线程。默认为False，表示不使用多线程。

    返回:
        np.ndarray or list: 图片的NumPy数组，如果输入是字符串，则返回单个数组；如果输入是元组列表，
        则返回NumPy数组的列表。

    错误:
        FileNotFoundError: 当指定的图片文件不存在时引发异常。
        ValueError: 当输入不是合法的MD5字符串或MD5字符串列表时引发异常。
    """

    if isinstance(md5, str):  # 如果输入是单个MD5字符串
        md5_list = [md5]
    elif isinstance(md5, list):  # 如果输入是MD5字符串列表
        md5_list = md5
    else:
        raise ValueError("Invalid input format. Expected string or list of strings for 'md5'.")

    images = []

    def read_image(md5):
        img_path = os.path.join(path or "./MangaImg", md5 + ".webp")

        try:
            img = Image.open(img_path)
            if type == 'numpy':
                img_np = np.array(img)
                return img_np
            else:
                return img
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Image file with MD5 '{md5}' not found at path: {img_path}.") from e

    if use_threading:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = executor.map(read_image, md5_list)

            for result in results:
                images.append(result)
    else:
        for md5 in md5_list:
            result = read_image(md5)
            images.append(result)

    if len(images) == 1:
        return images[0]
    else:
        return images



# 使用示例
# try:
#     md5_or_md5_list = "your_md5_string_here"  # 或者使用 [("md5_1",), ("md5_2",), ...] 来传入多个md5
#     image_path = "your_custom_image_folder_path_here"  # 如果有指定路径，可以传入指定路径
#
#     result = read_image_from_md5(md5_or_md5_list, image_path)
#     print(result)
# except Exception as e:
#     print("Error:", e)
def convert_first_and_to_where(query):
    """
    将查询语句中的第一个 "AND" 转换为 "WHERE"，如果已经存在 "WHERE" 则跳过替换。

    参数：
    - query（str）：查询语句。

    返回值：
    - new_query（str）：转换后的查询语句。
    """
    if " AND " in query:
        if " WHERE " not in query:
            new_query = query.replace(" AND ", " WHERE ", 1)
            return new_query
    return query


class MangaDbObj:
    def __init__(self, db_path=None):
        """
        初始化 MangaDbObj 对象，并连接到 SQLite 数据库。

        参数：
        - db_path：数据库文件路径，默认为当前路径下的 "Mangadb.db" 文件。

        注意事项：
        - 如果连接数据库出错，会打印错误信息。
        """
        if db_path is None:
            db_path = "Mangadb.db"  # 默认数据库路径为当前路径下的 Mangadb.db 文件

        # 连接到 SQLite 数据库，并且创建表
        try:
            self.conn = sqlite3.connect(db_path,timeout=4096, check_same_thread=False)
            self.cursor = self.conn.cursor()
            print("Successfully connected to the database")
            self.create_tables()
            self.release_multi_process_write()#开启多线程
        except sqlite3.Error as e:
            print(f"Error connecting to the database: {str(e)}")

    def create_tables(self):
        """
        创建数据库中的两个表：MangaPegeIndex 和 MangaBook。

        注意事项：
        - 如果创建表出错，会打印错误信息。
        """
        # 创建 MangaPegeIndex 表
        try:
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS MangaPegeIndex (
                    md5 TEXT PRIMARY KEY,
                    indexnum INTEGER,
                    phash TEXT
                )
            """)
            # print("MangaPegeIndex table created successfully")
        except sqlite3.Error as e:
            # print(f"Error creating MangaPegeIndex table: {str(e)}")
            pass


        try:
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS MangaBook (
                    gid INTEGER PRIMARY KEY,
                    path TEXT,
                    name TEXT,
                    title TEXT,
                    imgindex TEXT,
                    label TEXT,
                    is_deleted BOOL,
                    time TEXT
                )
            """)
            # print("MangaBook table created successfully")
        except sqlite3.Error as e:
            # print(f"Error creating MangaBook table: {str(e)}")
            pass

    def get_manga_pege_index_row_count(self):
        """
        查询 MangaPegeIndex 表的行数。
        """
        try:
            self.cursor.execute("SELECT COUNT(*) FROM MangaPegeIndex")
            row_count = self.cursor.fetchone()[0]
            return row_count
        except sqlite3.Error as e:
            print(f"Error querying row count for MangaPegeIndex table: {str(e)}")
            return None

    def get_manga_book_row_count(self):
        """
        查询 MangaBook 表的行数。
        """
        try:
            self.cursor.execute("SELECT COUNT(*) FROM MangaBook")
            row_count = self.cursor.fetchone()[0]
            return row_count
        except sqlite3.Error as e:
            print(f"Error querying row count for MangaBook table: {str(e)}")
            return None

    def commit_transaction(self):
        """
        提交事务。

        注意事项：
        - 如果提交事务出错，会打印错误信息。
        """
        try:
            self.conn.commit()
            print("Transaction committed successfully")
        except sqlite3.Error as e:
            print(f"Error committing transaction: {str(e)}")

    def rollback_transaction(self):
        """
        回滚事务。

        注意事项：
        - 如果回滚事务出错，会打印错误信息。
        """
        try:
            self.conn.rollback()
            print("Transaction rolled back successfully")
        except sqlite3.Error as e:
            print(f"Error rolling back transaction: {str(e)}")

    def release_multi_process_write(self):
        """
        解除多进程写操作的限制。

        注意事项：
        - 如果执行解除操作出错，会打印错误信息。
        """
        try:
            self.cursor.execute("PRAGMA journal_mode = wal")  # 设置日志模式为wal
            self.cursor.execute("PRAGMA synchronous = NORMAL")  # 设置同步模式为NORMAL
            print("Multi-process write restriction released")
        except sqlite3.Error as e:
            print(f"Error releasing multi-process write restriction: {str(e)}")

    def restore_single_process_write(self):
        """
        恢复单进程写操作模式。

        注意事项：
        - 如果执行恢复操作出错，会打印错误信息。
        """
        try:
            self.cursor.execute("PRAGMA journal_mode = delete")  # 设置日志模式为delete
            self.cursor.execute("PRAGMA synchronous = full")  # 设置同步模式为full
            print("Single-process write mode restored")
        except sqlite3.Error as e:
            print(f"Error restoring single-process write mode: {str(e)}")

    def get_all_manga_page_index(self):
        """
        查询并返回MangaPegeIndex表中的所有信息。

        返回值：
        - rows：包含查询结果的列表，每个元素是一个包含表中一行数据的元组。
        """
        try:
            self.cursor.execute("SELECT * FROM MangaPegeIndex")
            rows = self.cursor.fetchall()
            return rows
        except sqlite3.Error as e:
            print(f"Error executing query: {str(e)}")
            return []
    
    def insert_manga_book(self, *args):
        """
        向 MangaBook 表插入漫画信息。

        参数：
        - args：可以是元组列表形式 [(path1, name1, title1, imgindex1, label1, is_deleted1), (path2, name2, title2, imgindex2, label2, is_deleted2), ...]，
                或者多个单独的值形式 path1, name1, title1, imgindex1, label1, is_deleted1, path2, name2, title2, imgindex2, label2, is_deleted2, ...

        注意事项：
        - 如果插入出错，会打印错误信息。
        """
        # 向 MangaBook 表插入漫画信息
        if len(args) == 1 and isinstance(args[0], list):
            # 如果参数是元组列表形式 [(path1, name1, title1, imgindex1, label1, is_deleted1), (path2, name2, title2, imgindex2, label2, is_deleted2), ...]
            data = args[0]
        else:
            # 如果参数是多个单独的值形式 path1, name1, title1, imgindex1, label1, 
            # 1, path2, name2, title2, imgindex2, label2, is_deleted2, ...
            data = [args[i:i+6] for i in range(0, len(args), 6)]
        try:
            time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # 获取当前时间
            self.cursor.executemany("""
                INSERT INTO MangaBook (path, name, title, imgindex, label, is_deleted, time)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, [(path, name, title, imgindex, label, is_deleted, time) for path, name, title, imgindex, label, is_deleted in data])
            #self.conn.commit()
            print("Manga book inserted successfully")
        except sqlite3.Error as e:
            print(f"Error inserting manga book: {str(e)}")
    
    def increase_indexnum(self, *args):
        """
        废弃
        增加指定 MD5 的 indexnum 值。

        参数：
        - args：可以是元组列表形式 [(md51, md52, ...)]，或者多个单独的值形式 md51, md52, ...

        注意事项：
        - 如果增加出错，会打印错误信息。
        """
        # 增加指定 MD5 的 indexnum 值
        if len(args) == 1 and isinstance(args[0], list):
            # 如果参数是元组列表形式 [(md51, md52, ...)]
            md5_list = args[0]
        else:
            # 如果参数是多个单独的值形式 md51, md52, ...
            md5_list = args
        try:
            self.cursor.executemany("""
                UPDATE MangaPegeIndex
                SET indexnum = indexnum + 1
                WHERE md5 = ?
            """, [(md5,) for md5 in md5_list])
            #self.conn.commit()
            print("Indexnum increased successfully")
        except sqlite3.Error as e:
            print(f"Error increasing indexnum: {str(e)}")

    def decrease_indexnum(self, *args):
        """
        废弃
        减少指定 MD5 的 indexnum 值，不允许小于零。

        参数：
        - args：可以是元组列表形式 [(md51, md52, ...)]，或者多个单独的值形式 md51, md52, ...

        注意事项：
        - 如果减少出错，会打印错误信息。
        """
        # 减少指定 MD5 的 indexnum 值，不允许小于零
        if len(args) == 1 and isinstance(args[0], list):
            # 如果参数是元组列表形式 [(md51, md52, ...)]
            md5_list = args[0]
        else:
            # 如果参数是多个单独的值形式 md51, md52, ...
            md5_list = args
        try:
            self.cursor.executemany("""
                UPDATE MangaPegeIndex
                SET indexnum = indexnum - 1
                WHERE md5 = ? AND indexnum > 0
            """, [(md5,) for md5 in md5_list])
            #self.conn.commit()
            print("Indexnum decreased successfully")
        except sqlite3.Error as e:
            print(f"Error decreasing indexnum: {str(e)}")

    def update_pege_index(self, *args):
        """
        更新 MangaPegeIndex 表中的页信息。

        参数：
        - args：可以是元组列表形式 [(md5, phash), ...]，或者多个单独的值形式 md5, phash。

        注意事项：
        - 如果更新出错，会打印错误信息。
        """
        # 更新 MangaPegeIndex 表中的页信息
        if len(args) == 1 and isinstance(args[0], list):
            # 如果参数是元组列表形式 [(md5, phash), ...]
            data = args[0]
        else:
            # 如果参数是多个单独的值形式 md5, phash
            data = [args]

        try:
            for md5, phash in data:
                if md5 is None:
                    continue

                self.cursor.execute("""
                    SELECT indexnum FROM MangaPegeIndex WHERE md5 = ?
                """, (md5,))
                result = self.cursor.fetchone()

                if result is not None:
                    indexnum = result[0]
                    self.cursor.execute("""
                        UPDATE MangaPegeIndex
                        SET indexnum = ?
                        WHERE md5 = ?
                    """, (indexnum + 1, md5))
                else:
                    self.cursor.execute("""
                        INSERT INTO MangaPegeIndex (md5, indexnum, phash)
                        VALUES (?, 1, ?)
                    """, (md5, phash))

            #self.conn.commit()
            print("pege index updated successfully")
        except sqlite3.Error as e:
            print(f"Error updating pege index: {str(e)}")

    def decrease_indexnum(self, *args):
        """
        减少指定 MD5 的 indexnum 值，不允许小于零。

        参数：
        - args：可以是元组列表形式 [(md51, md52, ...)]，或者多个单独的值形式 md51, md52, ...

        注意事项：
        - 如果减少出错，会打印错误信息。
        - 如果 indexnum 小于等于零，会抛出异常。
        """
        # 减少指定 MD5 的 indexnum 值，不允许小于零
        if len(args) == 1 and isinstance(args[0], list):
            # 如果参数是元组列表形式 [(md51, md52, ...)]
            md5_list = args[0]
        else:
            # 如果参数是多个单独的值形式 md51, md52, ...
            md5_list = args
        try:
            for md5 in md5_list:
                self.cursor.execute("""
                    SELECT indexnum FROM MangaPegeIndex WHERE md5 = ?
                """, (md5,))
                result = self.cursor.fetchone()

                if result is not None:
                    indexnum = result[0]
                    if indexnum > 0:
                        self.cursor.execute("""
                            UPDATE MangaPegeIndex
                            SET indexnum = ?
                            WHERE md5 = ?
                        """, (indexnum - 1, md5))
                    else:
                        raise ValueError(f"Indexnum for MD5 '{md5}' cannot be less than zero")
                else:
                    raise ValueError(f"MD5 '{md5}' not found in MangaPegeIndex")

            #self.conn.commit()
            print("Indexnum decreased successfully")
        except sqlite3.Error as e:
            print(f"Error decreasing indexnum: {str(e)}")

    def check_duplicate_paths(self, paths):
        """
        检查压缩包路径列表中是否存在已经存在于数据库表 MangaBook 的路径记录，
        如果存在，则将其从列表中移除，返回符合条件的路径列表。

        参数：
        - paths（list）：压缩包路径列表。

        返回值：
        - valid_paths（list）：符合条件的路径列表。
        """
        try:
            valid_paths = []
            query = "SELECT path FROM MangaBook WHERE path IN ({})".format(",".join(["?"] * len(paths)))
            self.cursor.execute(query, paths)
            existing_paths = [row[0] for row in self.cursor.fetchall()]

            for path in paths:
                if path not in existing_paths:
                    valid_paths.append(path)

            return valid_paths

        except sqlite3.Error as e:
            print(f"Error checking duplicate paths: {str(e)}")
            return []
    #废弃
    def insert_pege_index(self, *args):
        """
        向 MangaPegeIndex 表插入页的信息。

        参数：
        - args：可以是元组列表形式 [(md5, indexnum, phash), ...]，或者多个单独的值形式 md5, indexnum, phash。

        注意事项：
        - 如果插入出错，会打印错误信息。
        """
        # 向 MangaPegeIndex 表插入页的信息
        if len(args) == 1 and isinstance(args[0], list):
            # 如果参数是元组列表形式 [(md5, indexnum, phash), ...]
            data = args[0]
        else:
            # 如果参数是多个单独的值形式 md5, indexnum, phash
            data = [args]
        try:
            self.cursor.executemany("""
                INSERT INTO MangaPegeIndex (md5, indexnum, phash)
                VALUES (?, ?, ?)
            """, data)
            #self.conn.commit()
            print("pege index inserted successfully")
        except sqlite3.Error as e:
            print(f"Error inserting pege index: {str(e)}")

    def get_zero_indexnum_md5(self):
        """
        获取所有 indexnum 为 0 的 md5 列表。

        返回值：
        - md5_list: 包含所有 indexnum 为 0 的 md5 的列表。

        注意事项：
        - 如果获取出错，会打印错误信息。
        """
        try:
            self.cursor.execute("""
                SELECT md5 FROM MangaPegeIndex WHERE indexnum = 0
            """)
            result = self.cursor.fetchall()
            md5_list = [row[0] for row in result]
            return md5_list
        except sqlite3.Error as e:
            print(f"Error getting zero indexnum md5 list: {str(e)}")
            return []


    def get_manga_books(self, pege=1, pege_size=100, order_by="gid", start_date=None, end_date=None, include_is_deleted=False, search_text=None, search_mode="exact", random_order=False, min_page=None, max_page=None):
        """
        获取漫画书籍的列表。

        参数：
        - pege（int）：页码，默认为 1。
        - pege_size（int）：每页记录数，默认为 100。
        - order_by（str）：排序字段，默认为 "gid"，可选字段有 "gid" 和 "time"。
        - start_date（str）：开始日期，格式为 "YYYY-MM-DD"，默认为 None。
        - end_date（str）：结束日期，格式为 "YYYY-MM-DD"，默认为 None。
        - include_is_deletedd（bool）：是否包含已删除的记录，默认为 False。
        - search_text（str）：搜索文本，默认为 None。
        - search_mode（str）：搜索模式，默认为 "exact"，可选值为 "exact" 和 "fuzzy"。
        - random_order（bool）：是否乱序，默认为 False。
        - min_page（int）：最小页数，默认为 None。
        - max_page（int）：最大页数，默认为 None。

        返回值：
        - manga_books（list）：漫画书籍列表，每个元素为一个字典，包含以下字段：
            - gid（int）：唯一标识符（主键）。
            - path（str）：路径。
            - name（str）：文件名。
            - title（str）：标题。
            - imgindex（dict）：页码和对应的 md5 值的字典。
            - label（dict）：标签的字典。
            - is_deleted（bool）：是否已删除。
            - time（str）：插入数据的时间。

        注意：
        - 当指定了 start_date 和 end_date 时，将按照时间范围进行筛选。
        - 当 include_is_deletedd 为 False 时，将排除已删除的记录。
        - 当 search_text 不为 None 时，将进行搜索过滤，可以选择精确模式或模糊模式。
            - 精确模式（exact）：搜索文本必须精确匹配 path、name、title 字段的一部分。
            - 模糊模式（fuzzy）：忽略标点符号和字符串位置，进行模糊匹配。
        - 当 random_order 为 True 时，将返回乱序的结果。
        - 当指定了 min_page 和 max_page 时，将按照页数范围进行筛选。

        异常：
        - 如果连接数据库失败或执行查询失败，将打印错误信息并返回空列表 []。
        """
        try:
            offset = (pege - 1) * pege_size

            query = f"""
                SELECT gid, path, name, title, imgindex, label, is_deleted, time
                FROM MangaBook
                """

            # 添加时间范围筛选条件
            if start_date and end_date:
                query += f" AND time BETWEEN '{start_date}' AND '{end_date}'"
            elif start_date:
                query += f" AND time >= '{start_date}'"
            elif end_date:
                query += f" AND time <= '{end_date}'"

            # 添加是否包含已删除的记录条件
            if not include_is_deleted:
                query += " AND is_deleted = 0"

            # 添加搜索条件
            if search_text:
                if search_mode == "exact":
                    query += f" AND (path LIKE '%{search_text}%' OR name LIKE '%{search_text}%' OR title LIKE '%{search_text}%')"
                elif search_mode == "fuzzy":
                    search_text = re.sub(r'[^\w\s]', '', search_text)  # 移除标点符号
                    query += f" AND (path LIKE '%{search_text}%' ESCAPE '\\' OR name LIKE '%{search_text}%' ESCAPE '\\' OR title LIKE '%{search_text}%')"



            # 添加页数范围筛选条件
            calculate_character_count = lambda n: n * 32 + 6 + 7 * (n - 1)
            if min_page is not None and max_page is not None:
                query += f" AND LENGTH(imgindex) >= {calculate_character_count(min_page)} AND LENGTH(imgindex) <= {calculate_character_count(max_page)}"
            elif min_page is not None:
                query += f" AND LENGTH(imgindex) >= {calculate_character_count(min_page)}"
            elif max_page is not None:
                query += f" AND LENGTH(imgindex) <= {calculate_character_count(max_page)}"

            

            # 添加乱序
            if random_order:
                query += " ORDER BY RANDOM()"
            else:
                # 添加排序方式，默认为倒序
                query += f" ORDER BY {order_by} DESC"

            # 添加分页限制
            query += f" LIMIT {pege_size} OFFSET {offset}"
            query=convert_first_and_to_where(query)#将第一个AND转换为WHERE
            print(query)
            self.cursor.execute(query)
            
            rows = self.cursor.fetchall()
            manga_books = []

            for row in rows:
                gid, path, name, title, imgindex, label, is_deleted, time = row
                if imgindex is None or imgindex == '':
                    imgindex="{}"
                if label is None or label == '': 
                    label='{}'
                imgindex = json.loads(imgindex)
                label = json.loads(label)

                manga_book = {
                    "gid": gid,
                    "path": path,
                    "name": name,
                    "title": title,
                    "imgindex": imgindex,#json
                    "label": label,#json
                    "is_deleted": bool(is_deleted),
                    "time": time
                }
                manga_books.append(manga_book)

            return manga_books

        except sqlite3.Error as e:
            print(f"Error retrieving manga books: {str(e)}")
            return []

    def compare_image_with_db(self, img):
        """
        计算一个图片的感知哈希值，并与数据库中的感知哈希值进行比较，按相似度排序输出。

        参数：
        - img：图片数据，可以是路径字符串、字节流、NumPy 数组、PIL 图像对象。

        返回值：
        - sorted_results：按相似度排序的查询结果，包含每个匹配项的元组，格式为 (md5, indexnum, phash, similarity)。
        """
        # 计算图片的感知哈希值
        phash_img = CalculateImgPhash(img)

        # 查询数据库中的所有信息
        try:
            self.cursor.execute("SELECT * FROM MangaPegeIndex")
            rows = self.cursor.fetchall()
        except sqlite3.Error as e:
            print(f"Error executing query: {str(e)}")
            return []

        # 定义计算相似度的函数
        def calculate_similarity(row):
            md5, indexnum, phash = row
            similarity = compute_similarity(imagehash.hex_to_hash(phash_img), imagehash.hex_to_hash(phash))
            return (md5, indexnum, phash, similarity)

        # 使用线程池并发计算相似度
        results = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            similarity_futures = [executor.submit(calculate_similarity, row) for row in rows]
            for future in concurrent.futures.as_completed(similarity_futures):
                result = future.result()
                results.append(result)

        # 按相似度排序结果
        sorted_results = sorted(results, key=lambda x: x[3], reverse=True)
        return sorted_results


#现在我要定义一个类方法，输入之前的process_images_in_archive中返回的image_info_list，或者image_info_list其中的一个元组。

# 示例用法
# db_obj = MangaDbObj()  # 在当前路径下创建或连接到 Mangadb.db
# db_obj.create_tables()  # 创建数据库表
# db_obj.close_connection()  # 关闭数据库连接

def transpose_tuple_list(tuple_list):
    """
    将元组列表进行转置。

    参数:
    - tuple_list: 元组列表，要求元组的元素个数相同。

    返回值:
    - transposed_list: 转置后的元组列表，每个元组的元素对应原列表的相应位置。

    示例:
    >>> tuple_list = [(j1, k1, l1), (j2, k2, l2), (j3, k3, l3)]
    >>> transposed_list = transpose_tuple_list(tuple_list)
    >>> print(transposed_list)
    [(j1, j2, j3), (k1, k2, k3), (l1, l2, l3)]
    """

    transposed_list = list(zip(*tuple_list))
    return transposed_list

def tuple_to_json(tuple_data):
    """
    将元组转化为 JSON 格式的字符串。

    参数：
    - tuple_data：待转换的元组。

    返回值：
    - json_data：转换后的 JSON 格式的字符串。

    输入: (123, 234, 345, 456)
    输出: {"pege1": 123, "pege2": 234, "pege3": 345, "pege4": 456}

    """
    if tuple_data is None:
        return
    json_data = json.dumps({"pege" + str(i+1): value for i, value in enumerate(tuple_data)})
    return json_data

def convert_to_tuple_list(lst):
    """
    将列表中的每个元素转换为包含单个元素的元组。

    参数：
    - lst：输入的列表。

    返回值：
    - tuple_lst：转换后的元组列表。

    示例：
    >>> lst = [k1, k2, ..., kn]
    >>> tuple_lst = convert_to_tuple_list(lst)
    >>> print(tuple_lst)
    [(k1,), (k2,), ..., (kn,)]
    """
    tuple_lst = [(item,) for item in lst]
    return tuple_lst
import os

def get_archive_paths(folder_path):
    """
    获取指定文件夹内所有的压缩包路径。

    参数：
    - folder_path：文件夹路径。

    返回值：
    - archive_paths：压缩包路径列表。

    示例：
    >>> folder_path = "/path/to/folder"
    >>> archive_paths = get_archive_paths(folder_path)
    >>> print(archive_paths)
    ["/path/to/folder/archive1.zip", "/path/to/folder/archive2.tar.gz", ...]
    """
    archive_paths = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path) and file_name.endswith(('.zip', '.tar', '.tar.gz', '.tar.bz2', '.tar.xz')):
            archive_paths.append(file_path)
    return archive_paths

def filter_image_zip_paths(zip_paths, use_multithreading=True):
    """
    筛选符合条件的压缩包路径列表。

    参数：
    - zip_paths: 压缩包路径列表。
    - use_multithreading: 是否使用多线程，默认为 True。

    返回值：
    - filtered_paths: 符合条件的压缩包路径列表。
    """

    filtered_paths = []

    def process_zip(zip_path):
        ext = os.path.splitext(zip_path)[1].lower()

        try:
            if ext == '.zip':
                with zipfile.ZipFile(zip_path, 'r') as zip_file:
                    for file_info in zip_file.infolist():
                        file_ext = os.path.splitext(file_info.filename)[1].lower()
                        if file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
                            filtered_paths.append(zip_path)
                            break

            elif ext == '.7z':
                with py7zr.SevenZipFile(zip_path, mode='r') as seven_zip_file:
                    for file_info in seven_zip_file.getnames():
                        file_ext = os.path.splitext(file_info)[1].lower()
                        if file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
                            filtered_paths.append(zip_path)
                            break

        except (OSError, zipfile.BadZipFile, py7zr.Bad7zFile):
            pass

    if use_multithreading:
        with ThreadPoolExecutor() as executor:
            executor.map(process_zip, zip_paths)
    else:
        for zip_path in zip_paths:
            process_zip(zip_path)

    return filtered_paths

def compute_similarity(phash1, phash2):
    distance = np.count_nonzero(phash1.hash != phash2.hash)
    normalized_distance = distance / max(phash1.hash.size, phash2.hash.size)
    return normalized_distance

# if __name__ == "__main__":
def debug1():#作用是导入漫画到数据库中去
    archive_path_list=get_archive_paths("Z:\\testmanga")#获取文件夹内所有压缩包的路径列表
    MangaDb=MangaDbObj()
    TestTime()
    #筛选符合条件的漫画
    archive_path_list2=filter_image_zip_paths(archive_path_list)#符合条件的压缩包
    archive_path_list3=MangaDb.check_duplicate_paths(archive_path_list2)#移除已经处理完成的压缩包
    TestTime()
    #插入单本漫画
    for archive_path in archive_path_list3:
        print(archive_path)
        image_info_list = process_images_in_archive(archive_path)#处理压缩包获得数据
        error, md5_hash, phash, img_data=transpose_tuple_list(image_info_list)
        file_name = os.path.basename(archive_path)#获取文件名
        imgindex=tuple_to_json(md5_hash)
        label=""
        try:
            #保存图片文件
            save_image_as_webp(img_data)
            # 数据库操作 漫画书
            MangaDb.insert_manga_book(archive_path,file_name,None,imgindex,label,False)#插入漫画书
            #数据库操作 漫画页
            md5_hash_phash_list=transpose_tuple_list(transpose_tuple_list(zip(md5_hash, phash)))
            MangaDb.update_pege_index(md5_hash_phash_list)
            #提交数据
            MangaDb.commit_transaction()
        except Exception:
            #回滚数据库
            MangaDb.rollback_transaction()

    TestTime()



def debug2():#作用是导入漫画到数据库中去
    archive_path_list=get_archive_paths("Z:\\testmanga")#获取文件夹内所有压缩包的路径列表
    MangaDb=MangaDbObj()
    TestTime()
    #筛选符合条件的漫画
    archive_path_list2=filter_image_zip_paths(archive_path_list)#符合条件的压缩包
    archive_path_list3=MangaDb.check_duplicate_paths(archive_path_list2)#移除已经处理完成的压缩包
    TestTime()
    #以上是获取任务
    for archive_path in archive_path_list3:
        print(archive_path)
        image_info_list = process_images_in_archive(archive_path)#处理压缩包获得数据 #cpu密集型
        error, md5_hash, phash, img_data=transpose_tuple_list(image_info_list)
        file_name = os.path.basename(archive_path)#获取文件名
        imgindex=tuple_to_json(md5_hash)
        label=""
        try:
            #保存图片文件
            save_image_as_webp(img_data)#内部自带多进程
            # 数据库操作 漫画书
            MangaDb.insert_manga_book(archive_path,file_name,None,imgindex,label,False) #插入漫画书
            #数据库操作 漫画页
            md5_hash_phash_list=transpose_tuple_list(transpose_tuple_list(zip(md5_hash, phash)))
            MangaDb.update_pege_index(md5_hash_phash_list)
            #提交数据
            MangaDb.commit_transaction()
        except Exception:
            #回滚数据库
            MangaDb.rollback_transaction()

    TestTime()


def count_elements(data):
    """
    计算给定数据结构中元素的个数。

    参数:
        data (str, dict, tuple, str, list): 输入的数据结构，可以是以下类型：
            - JSON字符串
            - 字典
            - 元组
            - 字符串
            - 列表

    返回:
        int: 元素的个数。

    错误:
        ValueError: 当输入为JSON字符串但格式无效时引发异常。

    注意事项:
        - 如果输入是JSON字符串，则将其解析为字典进行计算。
        - 对于字典、元组和列表，使用 `len()` 函数计算元素的个数。
        - 对于其他类型（字符串或单个元素），返回1。
    """
    # 方法实现的代码
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except ValueError:
            raise ValueError("Invalid JSON format")
    
    if isinstance(data, dict):
        return len(data)
    elif isinstance(data, (tuple, list)):
        return len(data)
    else:
        return 1



def process_archive(archive_path):
    try:
        MangaDb = MangaDbObj()
        image_info_list = process_images_in_archive(archive_path)
        error, md5_hash, phash, img_data = transpose_tuple_list(image_info_list)
        file_name = os.path.basename(archive_path)
        imgindex = tuple_to_json(md5_hash)
        label = ""
        print(archive_path)
        try:
            save_image_as_webp(img_data)
            MangaDb.insert_manga_book(archive_path, file_name, None, imgindex, label, False)
            global updata_manga
            if updata_manga!={}:
                put_text(archive_path+" "+file_name+" "+imgindex)
            md5_hash_phash_list = transpose_tuple_list(transpose_tuple_list(zip(md5_hash, phash)))
            MangaDb.update_pege_index(md5_hash_phash_list)
            MangaDb.commit_transaction()
        except Exception:
            MangaDb.rollback_transaction()
    except Exception as e:
        print(f"Error processing archive {archive_path}: {str(e)}")

def debug2(cpu_cores=0.25,path="Z:\\testmanga"):
    #多进程导入压缩包
    archive_path_list = get_archive_paths(path)
    MangaDb = MangaDbObj()
    MangaDb.release_multi_process_write()#解除多进程写模式

    TestTime()
    archive_path_list2 = filter_image_zip_paths(archive_path_list)
    archive_path_list3 = MangaDb.check_duplicate_paths(archive_path_list2)
    TestTime()

    total_cpu_cores = psutil.cpu_count()
    if cpu_cores is None:
        executor = concurrent.futures.ProcessPoolExecutor()
    elif cpu_cores <= 1:
        executor = concurrent.futures.ProcessPoolExecutor(max_workers=int(total_cpu_cores * cpu_cores))
    else:
        executor = concurrent.futures.ProcessPoolExecutor(max_workers=int(cpu_cores))

    with executor as executor:
        executor.map(process_archive, archive_path_list3)
    
    # MangaDb.release_multi_process_write()
    TestTime()

def read_images_from_folder(folder_path='./testimg'):
    """
    从指定文件夹中读取所有图片文件，并返回PIL图像对象的列表。

    参数：
    - folder_path：文件夹路径，默认为当前目录下的testimg文件夹。

    返回值：
    - images：PIL图像对象的列表。
    """
    images = []
    # 遍历文件夹中的文件
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        # 判断文件是否为图片文件
        if os.path.isfile(file_path) and any(file_path.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']):
            # 打开图片文件并添加到列表中
            with open(file_path, 'rb') as f:
                img_data = f.read()
            img = Image.open(BytesIO(img_data))
            images.append(img)
    return images

def separate_list_elements(input_list):
    '''
      使用列表推导式将每个元素单独组成一个列表
    '''
    separated_lists = [[item] for item in input_list]
    # 返回包含每个元素单独列表的输出列表
    return separated_lists

def display_title(title):
    """
    以PyWebIO的方式显示标题。

    参数：
    - title：标题文本。

    注意事项：
    - 标题将以大号字体显示在PyWebIO界面上。
    """
    html = f'<h1>{title}</h1>'
    put_html(html)


def display_subtitle(subtitle):
    """
    以PyWebIO的方式显示副标题。

    参数：
    - subtitle：副标题文本。

    注意事项：
    - 副标题将以中号字体显示在PyWebIO界面上。
    """
    html = f'<h2>{subtitle}</h2>'
    put_html(html)

def display_images(images,display_mod="onebyone"):
    """
    以PyWebIO的方式输出图像列表。

    参数：
    - images：图像列表，包含路径字符串、字节流、NumPy 数组、PIL 图像对象。
        - displaymod 呈现方式，默认为"onebyone"。
        None为For循环加载以表格的方式一次性呈现
        "onebyone"为以循环的方式一张一张的输出图片
    注意事项：
    - 图像将以网格形式显示在PyWebIO界面上。
    """

    # 创建图像网格
    image_grid = []

    # 处理每个图像
    for img in images:
        # 将不同格式的图像统一转换为PIL图像对象
        if isinstance(img, str):  # 如果是路径字符串
            with open(img, 'rb') as f:
                img_data = f.read()
            img = Image.open(BytesIO(img_data))
        elif isinstance(img, bytes):  # 如果是字节流
            img = Image.open(BytesIO(img))
        elif isinstance(img, np.ndarray):  # 如果是NumPy数组
            img = Image.fromarray(img)
        elif not isinstance(img, Image.Image):  # 如果不是PIL图像对象
            raise ValueError("Unsupported image format")
        
        if display_mod is None or display_mod == "":
            image_grid.append(put_image(img))
        elif display_mod == "onebyone":
            put_image(img)
        # 将图像添加到网格中
        

    if display_mod is None or display_mod == "":
        put_grid(separate_list_elements(image_grid),cell_widths=1000)
    # elif display_mod == "onebyone":
    #     pass
    #     put_image(img)

def display_image(img,display_mod="onebyone"):
    """
    以PyWebIO的方式输出图片。

    参数：
    - image：图像，可以是路径字符串、字节流、NumPy 数组、PIL 图像对象，也可以是以上的列表。
    - displaymod 呈现方式，仅输入列表时有用，默认为空。
        None为以表格的方式一次性呈现 
        "onebyone"为以循环的方式一张一张的输出图片，呈现速度快
    """

    # 将不同格式的图像统一转换为PIL图像对象
    if isinstance(img, str):  # 如果是路径字符串
        with open(img, 'rb') as f:
            img_data = f.read()
        img = Image.open(BytesIO(img_data))
    elif isinstance(img, bytes):  # 如果是字节流
        img = Image.open(BytesIO(img))
    elif isinstance(img, np.ndarray):  # 如果是NumPy数组
        img = Image.fromarray(img)
    elif isinstance(img, list):#如果是列表，转为列表模式
        display_images(img,display_mod=display_mod)
        return
    elif not isinstance(img, Image.Image):  # 如果不是PIL图像对象
        raise ValueError("Unsupported image format")
    put_image(img)

def sort_dictionary_by_key(dictionary):
    """
    对字典按照键值进行排序，并返回排序后的值的列表。

    参数：
    - dictionary：需要排序的字典，键为格式为 "pege1", "pege2", ... 的字符串，值为任意类型。

    返回值：
    排序后的值的列表。

    注意事项：
    - 如果字典为空，将返回一个空列表。
    - 如果字典中的键不符合格式 "pege1", "pege2", ... 的要求，将引发 ValueError 异常。
    """

    try:
        sorted_list = [value for key, value in sorted(dictionary.items(), key=lambda x: int(x[0][4:]))]
        return sorted_list
    except (TypeError, ValueError) as e:
        print(f"Error sorting dictionary by key: {str(e)}")
        return []
    

def display_manga_debug():
    "漫画页md5，json文本"
    clear()
    mangabook=input(label="漫画页md5，json文本(现在已经废弃)")
    try:
        #data = json.loads(mangabook)
        pass
        #display_manga({'pege1': '6398177adf374f9e5097b53f3971891b', 'pege2': '93e459aceea5caaf3dcc0d3ee98623e8', 'pege3': '8b2de452d5edbb5ef367896864542717', 'pege4': 'd186db4037eec249992db83f7666eb7e', 'pege5': 'dfa0de23ad9aee5437b2f322d1494a8a'})
    except Exception:
        pass

def display_manga(manga_book=None,display_menu=False,display_info_buttons=False):
    clear()
    if manga_book==None:
        global manga_book_go_app
        if manga_book_go_app =={}:
            return
        manga_book=manga_book_go_app
        manga_book_go_app={}

    if display_info_buttons==True:
        put_buttons(['详情页'], onclick=[lambda:display_manga_info(manga_book)])
    if display_menu==True:
        menu()
    images = read_image_from_md5(sort_dictionary_by_key(manga_book["imgindex"]))
    display_image(images,display_mod="onebyone")
    if display_info_buttons==True:
        menu()
    if display_menu==True:
        put_buttons(['详情页'], onclick=[lambda:display_manga_info(manga_book)])
    
def display_manga_info_debug():
    manga_book={"gid":1,"path":"22222222222222222222222222222222222222222","name":"name111111111111111111111111111111111111111","title":"title111111111111111111111111111111111","imgindex":{'pege1': '6398177adf374f9e5097b53f3971891b', 'pege2': '93e459aceea5caaf3dcc0d3ee98623e8', 'pege3': '8b2de452d5edbb5ef367896864542717', 'pege4': 'd186db4037eec249992db83f7666eb7e', 'pege5': 'dfa0de23ad9aee5437b2f322d1494a8a'},"label":"","is_deleted":False,"time":"time1"}
    display_manga_info(manga_book)

def display_manga_info(manga_book=None):
    """
    以PyWebIO的方式输出漫画信息。

    参数：

    - manga_book：漫画书籍
        gid（int）：唯一标识符（主键）。
        path（str）：路径。
        name（str）：文件名。
        title（str）：标题。
        imgindex（dict）：页码和对应的 md5 值的字典。
        label（dict）：标签的字典。
        is_deleted（bool）：是否已删除。
        time（str）：插入数据的时间
    注意：
        如果输入的manga_book为空，将从全局变量manga_book_go_app获取漫画书，重置全局变量manga_book为{}。
    """
    if manga_book is None:
        global manga_book_go_app
        manga_book = manga_book_go_app
        manga_book_go_app={}
    if manga_book == {} or manga_book is None:
        return
    def create_onclick_function(book):
        global manga_book_go_app
        manga_book_go_app=book
        go_app('display_manga')
        #return lambda _: create_onclick_function_go_app(book)
    # def create_onclick_function_go_app(book):#启动另外一个的界面来阅读漫画
    #     global manga_book_go_app
    #     manga_book_go_app=book
    #     go_app('display_manga')
    #     return 

    clear()
    if not(manga_book["title"] == "" or manga_book["title"] is None):
        display_title(manga_book["title"])
    else:
        display_title(manga_book["name"])
    display_image(read_image_from_md5(manga_book["imgindex"]["pege1"]))
    if not(manga_book["title"] == "" or manga_book["title"] is None):
        display_title(manga_book["title"])
    else:
        display_title(manga_book["name"])
    put_buttons(['阅读'], onclick=[lambda:create_onclick_function(manga_book)])

    put_text("上传时间："+manga_book["time"])
    put_text("副标题："+manga_book["name"])
    put_text("页数:"+str(count_elements(manga_book["imgindex"])))
    if manga_book["is_deleted"]:
        put_text("状态：被删除")
    display_subtitle("标签")
    put_text("标签搜索功能在未来上线")
    display_subtitle("缩略图")
    put_text("缩略图功能会在未来上线")

    #put_buttons(['阅读'], onclick=[lambda:display_manga(manga_book)])

def debug_main():
    clear()
    put_buttons(['漫画浏览界面'], [lambda: go_app('display_manga_debug')])
    pass

def menu():
    put_buttons(['首页','搜索','随机','导入','信息'], onclick=[lambda: go_app("web_mian"),
                                                     lambda: go_app("select_manga_books"),
                                                     lambda: go_app("random_manga_books"),
                                                     lambda: go_app("inputmanga"),
                                                     lambda: go_app("sysinfo")])

def web_mian():
    clear()
    display_title("Manga仓库")
    menu()


def mian():
    web_mian()
    MangaDb = MangaDbObj()
    select_manga_books(page=1)
    pass

def display_manga_books(manga_books, display_mode="onebyone"):
    web_mian()
    manga_books
    display_lists=[]
    
    def create_onclick_function(book):
        return lambda _: create_onclick_function_go_app(book)
    def create_onclick_function_go_app(book):#启动另外一个的界面来阅读信息
        global manga_book_go_app
        manga_book_go_app=book
        go_app('display_manga_info')
        return 
    
    for manga_book in manga_books:
        if manga_book["title"] == "" and manga_book["title"] is None:
            manga_book["title"] = manga_book["name"]
        print("加载中")
        display_lists.append([span(manga_book["title"], col=3)])

        display_lists.append([span(put_image(generate_thumbnail(read_image_from_md5(manga_book["imgindex"]["pege1"]),width=300),height="100%",width="100%"),row=2), span("页数:"+str(count_elements(manga_book["imgindex"])), col=2)])
        display_lists.append([span(put_buttons(['详情'], onclick=create_onclick_function(manga_book)), col=2)])
        display_lists.append(["md5:" + manga_book["imgindex"]["pege1"], "id:" + str(manga_book["gid"]), manga_book["time"]])
        display_lists.append([span(" "*240, col=3)])

        if display_mode == "onebyone":
            put_table(display_lists, header=[span(manga_book["name"], col=3)])
            display_lists = []
            pass
    if display_mode=="onebyone":
        pass
    else:
        put_table(display_lists)
    

def select_manga_books(page=0, page_size=25, order_by="gid", start_date=None, end_date=None, include_is_deleted=False, search_text=None, search_mode="exact",random_order=False, min_page=None, max_page=None):
    MangaDb = MangaDbObj()
    def display_manga_books_wrapper(page):
        clear()
        manga_books = MangaDb.get_manga_books(pege=page, pege_size=page_size, order_by=order_by, start_date=start_date, end_date=end_date, include_is_deleted=include_is_deleted, search_text=search_text, search_mode=search_mode,random_order=random_order, min_page=min_page, max_page=max_page)
        display_manga_books(manga_books, display_mode="onebyone")
        if page == 0:
            put_buttons(["下一页"], onclick=lambda _: display_manga_books_wrapper(page+1))
        else:
            put_buttons(["上一页", "下一页"], onclick=lambda btn: display_manga_books_wrapper(page+1) if btn == "下一页" else display_manga_books_wrapper(page-1))

    if page == 0 or page == 1:
        clear()
        web_mian()
        if page==0:
            info = input_group("搜索", [
                input('关键字', name='search_text'),
                input('开始时间', name='start_time', type='date'),
                input('结束时间', name='end_time', type='date'),
                input('最少页数', name='min_pages', type='number'),
                input('最多页数', name='max_pages', type='number'),
                checkbox(options=['模糊搜索', '包含已删除'], name="checkbox_values")
            ])
            include_is_deleted = False
            search_mode = "exact"
            print(info)
            if info["search_text"] == "" or info['start_time'] is None:
                info[search_text] = None
            if info['start_time'] == "" or info['start_time'] is None:
                info['start_time'] = None
            if info['end_time'] == "" or info['end_time'] is None:
                info['end_time'] = None
            if info['min_pages'] == "" or info['min_pages'] is None:
                info['min_pages'] = None
            if info['max_pages'] == "" or info['max_pages'] is None:
                info['max_pages'] = None
            if '模糊搜索' in info["checkbox_values"]:
                search_mode = "fuzzy"
            else:
                search_mode = "exact"
            if '包含已删除' in info["checkbox_values"]:
                pass
            else:
                include_is_deleted = True
                page=1
            manga_books = MangaDb.get_manga_books(pege=1, pege_size=page_size, order_by="gid", start_date=info['start_time'], end_date=info['end_time'], include_is_deleted=include_is_deleted, search_text=info["search_text"], search_mode=search_mode,random_order=random_order, min_page=info["min_pages"], max_page=info["max_pages"])
        else:
            manga_books = MangaDb.get_manga_books(pege=1, pege_size=page_size, order_by="gid",random_order=random_order) 
        display_manga_books(manga_books, display_mode="onebyone")
        if page == 1:
            put_buttons(["下一页"], onclick=lambda _: display_manga_books_wrapper(page+1))
    else:
        display_manga_books_wrapper(page)


def random_manga_books():
    select_manga_books(page=1,random_order=True)
    pass

def index():
    mian()
    pass

def inputmanga():
    global updata_manga
    try:
        if updata_manga == {}:
            updata_manga="working"
            path = input("漫画库路径，导入时不要关闭页面。")
            updata_manga=path
            debug2(path=path)
            display_title("导入完成")
            updata_manga={}
        else:
            display_title("已经有一个导入进程在运行中，请等待操作完成。")
            put_text(str(updata_manga))
        pass
    except:
        updata_manga={}

import os
import concurrent.futures

def get_file_size(file_path):
    return os.path.getsize(file_path)

def get_folder_size(folder_path):
    total_size = 0

    with concurrent.futures.ThreadPoolExecutor() as thread_executor:
        with concurrent.futures.ProcessPoolExecutor() as process_executor:
            for path, dirs, files in os.walk(folder_path):
                file_paths = [os.path.join(path, f) for f in files]
                file_sizes = list(thread_executor.map(get_file_size, file_paths))
                total_size += sum(process_executor.map(lambda x: x, file_sizes))
    if total_size < 1024:
        size = f"{total_size} bytes"
    elif total_size < 1024 * 1024:
        size = f"{total_size/1024:.2f} KB"
    elif total_size < 1024 * 1024 * 1024:
        size = f"{total_size/1024/1024:.2f} MB"
    else:
        size = f"{total_size/1024/1024/1024:.2f} GB"
    return size

def sysinfo():
    def get_mangas_size(folder_path):
        folder_size=get_folder_size(folder_path)
        put_text("漫画占用空间：" + str(folder_size))
    folder_path = "./MangaImg"
    MangaDb = MangaDbObj()
    display_title("漫画库信息")
    put_text("存储的漫画本数：" + str(MangaDb.get_manga_book_row_count()))
    put_text("存储的漫画页总数：" + str(MangaDb.get_manga_pege_index_row_count()))
    put_buttons(["统计占用空间"], onclick=lambda _: get_mangas_size(folder_path))

global manga_book_go_app
global updata_manga
manga_book_go_app={} # 传递打开漫画界面时的参数
updata_manga={} # 导入漫画时，占位符，导入完成后恢复

if __name__ == '__main__':
    start_server([index,select_manga_books,
                  display_manga_info,
                  display_manga,
                  random_manga_books,
                  inputmanga,
                  sysinfo], port=8089)
