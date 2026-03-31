"""
邮件发送工具 — 通过 QQ 邮箱 SMTP 发送通知邮件

用法:
    from sendmail import send_notification
    send_notification(
        subject='训练完成',
        body='结果摘要...',
        attachments=['path/to/image.png'],
    )
"""

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from pathlib import Path

SMTP_SERVER = 'smtp.qq.com'
SMTP_PORT = 465
SENDER_EMAIL = '2861173454@qq.com'
SENDER_AUTH = 'wivoqbpgpsyudgcd'
RECEIVER_EMAIL = '2861173454@qq.com'


def send_notification(subject, body, attachments=None, receiver=None):
    """发送通知邮件，可附带多个文件附件。"""
    if receiver is None:
        receiver = RECEIVER_EMAIL

    msg = MIMEMultipart()
    msg['From'] = SENDER_EMAIL
    msg['To'] = receiver
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain', 'utf-8'))

    if attachments:
        for filepath in attachments:
            p = Path(filepath)
            if not p.exists():
                print(f'[Email] 附件不存在，跳过: {p}')
                continue
            with open(p, 'rb') as f:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(f.read())
            encoders.encode_base64(part)
            part.add_header(
                'Content-Disposition',
                f'attachment; filename="{p.name}"',
            )
            msg.attach(part)

    with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as server:
        server.login(SENDER_EMAIL, SENDER_AUTH)
        server.sendmail(SENDER_EMAIL, receiver, msg.as_string())

    print(f'[Email] 邮件已发送至 {receiver}')


if __name__ == '__main__':
    send_notification(
        subject='测试邮件 — IMDD Transformer 项目',
        body='这是一封测试邮件，用于验证 QQ 邮箱 SMTP 发送功能。',
    )
