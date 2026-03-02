import re
import hashlib

def clean_text(text):
    """إزالة المسافات الزائدة والرموز غير المرغوب فيها"""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def hash_company_name(name):
    """توليد معرف فريد للشركة (اختياري)"""
    return hashlib.md5(name.encode()).hexdigest()[:8]

def parse_retention_period(text):
    """تحويل نص مدة الاحتفاظ إلى عدد سنوات (تقريبي)"""
    text = text.lower()
    years = re.findall(r'(\d+)\s*year', text)
    if years:
        return int(years[0])
    months = re.findall(r'(\d+)\s*month', text)
    if months:
        return int(months[0]) / 12.0
    return 1.0  # افتراضي
