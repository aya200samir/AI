import re
from transformers import pipeline

class EntityExtractor:
    def __init__(self):
        # نموذج NER عام (يمكن استبداله بنموذج مخصص)
        self.ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
    
    def extract_entities(self, text):
        """استخراج الكيانات الأساسية من النص"""
        entities = {}
        
        # 1. اسم الشركة (عادة في البداية)
        company_match = re.search(r'([A-Z][A-Za-z\s]+(Ltd|Limited|LLC|Inc))', text)
        entities['company_name'] = company_match.group(1) if company_match else "Unknown"
        
        # 2. مسؤول حماية البيانات DPO
        dpo_match = re.search(r'Data Protection Officer[:\s]+([A-Z][a-z]+ [A-Z][a-z]+)', text, re.IGNORECASE)
        entities['dpo'] = dpo_match.group(1) if dpo_match else None
        
        # 3. أنواع البيانات (باستخدام NER)
        ner_results = self.ner(text[:2000])
        entities['data_types'] = list(set([ent['word'] for ent in ner_results if ent['entity_group'] in ['PER', 'LOC']]))  # تبسيط
        
        # 4. مدة الاحتفاظ
        retention_match = re.search(r'retain.*?for (\d+)\s*(year|month|day)', text, re.IGNORECASE)
        if retention_match:
            num = retention_match.group(1)
            unit = retention_match.group(2)
            entities['retention_period'] = f"{num} {unit}"
        else:
            entities['retention_period'] = "Not specified"
        
        # 5. نقل البيانات خارج الاتحاد الأوروبي
        entities['data_transfer_outside_eea'] = bool(re.search(r'transfer.*(outside|third country|international)', text, re.IGNORECASE))
        
        # 6. الحق في النسيان
        entities['right_to_be_forgotten'] = bool(re.search(r'right to (be forgotten|erasure)', text, re.IGNORECASE))
        
        return entities
