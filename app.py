{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "e73944c3-69c7-4e73-bde7-f4ce39736a4c",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "C:\\ProgramData\\anaconda3\\python.exe\n"
     ]
    }
   ],
   "source": [
    "import sys\n",
    "print(sys.executable)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "d45b8691-6717-4e03-b1a0-bb4463a3e673",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Done!\n"
     ]
    }
   ],
   "source": [
    "import sys\n",
    "import subprocess\n",
    "\n",
    "subprocess.check_call([\n",
    "    sys.executable, \"-m\", \"pip\", \"install\",\n",
    "    \"sentence-transformers\", \"pdfplumber\", \"torch\", \"scikit-learn\"\n",
    "])\n",
    "\n",
    "print(\"Done!\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "6036c958-28f5-4e53-a3e1-7a8264c533e7",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "جاري تحميل النموذج...\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "50537544f9094d308e9958f869ed76f1",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Loading weights:   0%|          | 0/199 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\u001b[1mBertModel LOAD REPORT\u001b[0m from: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2\n",
      "Key                     | Status     |  | \n",
      "------------------------+------------+--+-\n",
      "embeddings.position_ids | UNEXPECTED |  | \n",
      "\n",
      "\u001b[3mNotes:\n",
      "- UNEXPECTED\u001b[3m\t:can be ignored when loading from different task/architecture; not ok if you expect identical arch.\u001b[0m\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "النموذج جاهز!\n"
     ]
    }
   ],
   "source": [
    "from sentence_transformers import SentenceTransformer\n",
    "\n",
    "print(\"جاري تحميل النموذج...\")\n",
    "model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')\n",
    "print(\"النموذج جاهز!\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "0eb959fe-7079-49c8-9830-fb9414c140e1",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "تم إنشاء الفولدر: C:\\privacy_project\n",
      "ضعي ملفاتك فيه وشغلي الـ Cell تاني\n"
     ]
    }
   ],
   "source": [
    "import os\n",
    "\n",
    "# ضعي مسار الفولدر اللي فيه ملفاتك\n",
    "FOLDER = r\"C:\\privacy_project\"\n",
    "\n",
    "# تأكدي إن الفولدر موجود\n",
    "if not os.path.exists(FOLDER):\n",
    "    os.makedirs(FOLDER)\n",
    "    print(f\"تم إنشاء الفولدر: {FOLDER}\")\n",
    "    print(\"ضعي ملفاتك فيه وشغلي الـ Cell تاني\")\n",
    "else:\n",
    "    law_files = []\n",
    "    policy_files = []\n",
    "    keywords_law = ['قانون','law','151','لائحة','regulation','شرح']\n",
    "\n",
    "    for fname in os.listdir(FOLDER):\n",
    "        if fname.endswith(\".pdf\"):\n",
    "            full_path = os.path.join(FOLDER, fname)\n",
    "            if any(k in fname.lower() for k in keywords_law):\n",
    "                law_files.append(full_path)\n",
    "                print(f\"قانون: {fname}\")\n",
    "            else:\n",
    "                policy_files.append(full_path)\n",
    "                print(f\"سياسة: {fname}\")\n",
    "\n",
    "    print(f\"\\nملفات القانون: {len(law_files)}\")\n",
    "    print(f\"سياسات الخصوصية: {len(policy_files)}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "7fcf843c-b0cf-4a7e-b731-1ae4533597df",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "مكان الـ notebook: C:\\Users\\Admin\n",
      "\n",
      "الملفات الموجودة:\n",
      "----------------------------------------\n",
      "PDF: 23andme-penalty-notice.pdf\n",
      "PDF: 2410.03925v1.pdf\n",
      "PDF: California Consumer Privacy Act.pdf\n",
      "PDF: capita-plc-and-cpsl-monetary-penalty-notice.pdf\n",
      "PDF: ccpa_statute.pdf\n",
      "PDF: gdpr.pdf\n",
      "PDF: google_privacy_policy_en.pdf\n",
      "PDF: lastpass-uk-ltd-penalty-notice.pdf\n",
      "PDF: privacy policy NAB.pdf\n",
      "PDF: Privacy policy Spotify .pdf\n",
      "PDF: اللائحة التنفيذية لقانون حماية البيانات.pdf\n",
      "PDF: خصوصية البيانات وحماية البيانات.pdf\n",
      "PDF: دليل+نظام+حماية+البيانات+الشخصية+HANDBOOK_AR_ (1).pdf\n",
      "PDF: قانون حماية البيانات الشخصية .pdf\n",
      "PDF: مبادئ الأمم المتحدة العالمية __بشأن سالمة المعلومات.pdf\n"
     ]
    }
   ],
   "source": [
    "import os\n",
    "\n",
    "# هنشوف الملفات في نفس مكان الـ notebook\n",
    "current_folder = os.getcwd()\n",
    "print(f\"مكان الـ notebook: {current_folder}\")\n",
    "print(\"\\nالملفات الموجودة:\")\n",
    "print(\"-\" * 40)\n",
    "\n",
    "for fname in os.listdir(current_folder):\n",
    "    if fname.endswith(\".pdf\"):\n",
    "        print(f\"PDF: {fname}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "0b1ac7c5-f28a-48f2-a21d-cbb576b291a2",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "جاري قراءة ملفات القانون...\n",
      "  قانون: قانون حماية البيانات الشخصية .pdf — 0 فقرة\n",
      "  قانون: اللائحة التنفيذية لقانون حماية البيانات.pdf — 621 فقرة\n",
      "  قانون: دليل+نظام+حماية+البيانات+الشخصية+HANDBOOK_AR_ (1).pdf — 618 فقرة\n",
      "  قانون: خصوصية البيانات وحماية البيانات.pdf — 437 فقرة\n",
      "  قانون: مبادئ الأمم المتحدة العالمية __بشأن سالمة المعلومات.pdf — 586 فقرة\n",
      "  قانون: gdpr.pdf — 2183 فقرة\n",
      "  قانون: California Consumer Privacy Act.pdf — 1960 فقرة\n",
      "  قانون: ccpa_statute.pdf — 1960 فقرة\n",
      "  قانون: 2410.03925v1.pdf — 568 فقرة\n",
      "\n",
      "جاري قراءة ملفات العقوبات...\n",
      "  عقوبة: 23andme-penalty-notice.pdf — 3852 فقرة\n",
      "  عقوبة: capita-plc-and-cpsl-monetary-penalty-notice.pdf — 3296 فقرة\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Could not get FontBBox from font descriptor because None cannot be parsed as 4 floats\n",
      "Could not get FontBBox from font descriptor because None cannot be parsed as 4 floats\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "  عقوبة: lastpass-uk-ltd-penalty-notice.pdf — 2219 فقرة\n",
      "\n",
      "جاري قراءة سياسات الخصوصية...\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Could not get FontBBox from font descriptor because None cannot be parsed as 4 floats\n",
      "Could not get FontBBox from font descriptor because None cannot be parsed as 4 floats\n",
      "Could not get FontBBox from font descriptor because None cannot be parsed as 4 floats\n",
      "Could not get FontBBox from font descriptor because None cannot be parsed as 4 floats\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "  سياسة: google_privacy_policy_en.pdf — 598 فقرة\n",
      "  سياسة: Privacy policy Spotify .pdf — 373 فقرة\n",
      "  سياسة: privacy policy NAB.pdf — 35 فقرة\n",
      "\n",
      "========================================\n",
      "فقرات القانون: 8933\n",
      "فقرات العقوبات: 9367\n",
      "سياسات جاهزة: 3\n",
      "تم القراءة!\n"
     ]
    }
   ],
   "source": [
    "import pdfplumber\n",
    "import os\n",
    "\n",
    "current_folder = os.getcwd()\n",
    "\n",
    "# تصنيف الملفات\n",
    "law_files = [\n",
    "    \"قانون حماية البيانات الشخصية .pdf\",\n",
    "    \"اللائحة التنفيذية لقانون حماية البيانات.pdf\",\n",
    "    \"دليل+نظام+حماية+البيانات+الشخصية+HANDBOOK_AR_ (1).pdf\",\n",
    "    \"خصوصية البيانات وحماية البيانات.pdf\",\n",
    "    \"مبادئ الأمم المتحدة العالمية __بشأن سالمة المعلومات.pdf\",\n",
    "    \"gdpr.pdf\",\n",
    "    \"California Consumer Privacy Act.pdf\",\n",
    "    \"ccpa_statute.pdf\",\n",
    "    \"2410.03925v1.pdf\"\n",
    "]\n",
    "\n",
    "penalty_files = [\n",
    "    \"23andme-penalty-notice.pdf\",\n",
    "    \"capita-plc-and-cpsl-monetary-penalty-notice.pdf\",\n",
    "    \"lastpass-uk-ltd-penalty-notice.pdf\"\n",
    "]\n",
    "\n",
    "policy_files = [\n",
    "    \"google_privacy_policy_en.pdf\",\n",
    "    \"Privacy policy Spotify .pdf\",\n",
    "    \"privacy policy NAB.pdf\"\n",
    "]\n",
    "\n",
    "def read_pdf(filepath):\n",
    "    texts = []\n",
    "    try:\n",
    "        with pdfplumber.open(filepath) as pdf:\n",
    "            for page in pdf.pages:\n",
    "                text = page.extract_text()\n",
    "                if text:\n",
    "                    paragraphs = [\n",
    "                        p.strip() for p in text.split('\\n') \n",
    "                        if len(p.strip()) > 40\n",
    "                    ]\n",
    "                    texts.extend(paragraphs)\n",
    "    except Exception as e:\n",
    "        print(f\"خطأ في قراءة {filepath}: {e}\")\n",
    "    return texts\n",
    "\n",
    "# قراءة كل الملفات\n",
    "print(\"جاري قراءة ملفات القانون...\")\n",
    "law_texts = []\n",
    "for f in law_files:\n",
    "    path = os.path.join(current_folder, f)\n",
    "    if os.path.exists(path):\n",
    "        texts = read_pdf(path)\n",
    "        law_texts.extend(texts)\n",
    "        print(f\"  قانون: {f} — {len(texts)} فقرة\")\n",
    "\n",
    "print(f\"\\nجاري قراءة ملفات العقوبات...\")\n",
    "penalty_texts = []\n",
    "for f in penalty_files:\n",
    "    path = os.path.join(current_folder, f)\n",
    "    if os.path.exists(path):\n",
    "        texts = read_pdf(path)\n",
    "        penalty_texts.extend(texts)\n",
    "        print(f\"  عقوبة: {f} — {len(texts)} فقرة\")\n",
    "\n",
    "print(f\"\\nجاري قراءة سياسات الخصوصية...\")\n",
    "policies = {}\n",
    "for f in policy_files:\n",
    "    path = os.path.join(current_folder, f)\n",
    "    if os.path.exists(path):\n",
    "        texts = read_pdf(path)\n",
    "        policies[f] = texts\n",
    "        print(f\"  سياسة: {f} — {len(texts)} فقرة\")\n",
    "\n",
    "print(f\"\\n{'='*40}\")\n",
    "print(f\"فقرات القانون: {len(law_texts)}\")\n",
    "print(f\"فقرات العقوبات: {len(penalty_texts)}\")\n",
    "print(f\"سياسات جاهزة: {len(policies)}\")\n",
    "print(\"تم القراءة!\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "f4dba84b-0d4d-4508-8e0f-dce3252a226e",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "جاري تحميل النموذج...\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "3daef2d25eac4905b1a2acc15303d344",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Loading weights:   0%|          | 0/199 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\u001b[1mBertModel LOAD REPORT\u001b[0m from: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2\n",
      "Key                     | Status     |  | \n",
      "------------------------+------------+--+-\n",
      "embeddings.position_ids | UNEXPECTED |  | \n",
      "\n",
      "\u001b[3mNotes:\n",
      "- UNEXPECTED\u001b[3m\t:can be ignored when loading from different task/architecture; not ok if you expect identical arch.\u001b[0m\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "النموذج جاهز!\n",
      "\n",
      "جاري تحويل 8933 فقرة قانونية...\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "8fbee66a0fc1416f9706d71e2d60cf1d",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Batches:   0%|          | 0/280 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "جاري تحويل 9367 فقرة عقوبات...\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "59f0432e0f1948c38f0095bf70aa19f4",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Batches:   0%|          | 0/293 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "جاري تحويل السياسات...\n",
      "  google_privacy_policy_en.pdf...\n",
      "  Privacy policy Spotify .pdf...\n",
      "  privacy policy NAB.pdf...\n",
      "\n",
      "========================================\n",
      "vectors القانون: (8933, 384)\n",
      "vectors العقوبات: (9367, 384)\n",
      "السياسات المحولة: 3\n",
      "تم!\n"
     ]
    }
   ],
   "source": [
    "from sentence_transformers import SentenceTransformer\n",
    "import numpy as np\n",
    "\n",
    "print(\"جاري تحميل النموذج...\")\n",
    "model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')\n",
    "print(\"النموذج جاهز!\")\n",
    "\n",
    "# تحويل القانون لـ vectors\n",
    "print(f\"\\nجاري تحويل {len(law_texts)} فقرة قانونية...\")\n",
    "law_vectors = model.encode(\n",
    "    law_texts, \n",
    "    show_progress_bar=True,\n",
    "    batch_size=32\n",
    ")\n",
    "\n",
    "# تحويل العقوبات لـ vectors\n",
    "print(f\"\\nجاري تحويل {len(penalty_texts)} فقرة عقوبات...\")\n",
    "penalty_vectors = model.encode(\n",
    "    penalty_texts,\n",
    "    show_progress_bar=True,\n",
    "    batch_size=32\n",
    ")\n",
    "\n",
    "# تحويل السياسات لـ vectors\n",
    "print(\"\\nجاري تحويل السياسات...\")\n",
    "policy_vectors = {}\n",
    "for fname, texts in policies.items():\n",
    "    print(f\"  {fname}...\")\n",
    "    policy_vectors[fname] = model.encode(\n",
    "        texts,\n",
    "        show_progress_bar=False,\n",
    "        batch_size=32\n",
    "    )\n",
    "\n",
    "print(\"\\n\" + \"=\"*40)\n",
    "print(f\"vectors القانون: {law_vectors.shape}\")\n",
    "print(f\"vectors العقوبات: {penalty_vectors.shape}\")\n",
    "print(f\"السياسات المحولة: {len(policy_vectors)}\")\n",
    "print(\"تم!\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "bfc283bf-1461-42cf-83f4-0b547f3497ab",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "جاري بناء Dataset التدريب...\n",
      "========================================\n",
      "استخراج امثلة الامتثال من القانون...\n",
      "استخراج امثلة المخالفات من العقوبات...\n",
      "========================================\n",
      "Dataset جاهز!\n",
      "اجمالي الامثلة: 187\n",
      "امتثال (1): 88\n",
      "مخالفة (0): 99\n"
     ]
    }
   ],
   "source": [
    "from sklearn.metrics.pairwise import cosine_similarity\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "\n",
    "questions_db = [\n",
    "    {\"id\": 1,  \"text\": \"وضوح هوية المتحكم وبيانات الاتصال به والممثل المحلي في مصر\"},\n",
    "    {\"id\": 2,  \"text\": \"الأساس القانوني للمعالجة لكل غرض موافقة عقد التزام قانوني\"},\n",
    "    {\"id\": 3,  \"text\": \"الموافقة المسبقة الصريحة الحرة المستنيرة قبل جمع البيانات\"},\n",
    "    {\"id\": 4,  \"text\": \"تحديد أغراض استخدام البيانات بدقة وبشكل غير مضلل\"},\n",
    "    {\"id\": 5,  \"text\": \"الحد الأدنى من البيانات جمع أقل كمية ضرورية فقط\"},\n",
    "    {\"id\": 6,  \"text\": \"حق الوصول للبيانات والحصول على نسخة منها\"},\n",
    "    {\"id\": 7,  \"text\": \"حق التصحيح وتعديل البيانات غير الدقيقة\"},\n",
    "    {\"id\": 8,  \"text\": \"حق المحو والنسيان وحذف البيانات نهائيا\"},\n",
    "    {\"id\": 9,  \"text\": \"حق سحب الموافقة في اي وقت بسهولة\"},\n",
    "    {\"id\": 10, \"text\": \"حق الاعتراض على المعالجة لاغراض التسويق والبروفايل\"},\n",
    "    {\"id\": 11, \"text\": \"الافصاح عن مشاركة البيانات مع اطراف ثالثة وفئاتهم\"},\n",
    "    {\"id\": 12, \"text\": \"ضمانات الطرف الثالث والتزامه بنفس مستوى حماية البيانات\"},\n",
    "    {\"id\": 13, \"text\": \"نقل البيانات عبر الحدود بترخيص رسمي وضمانات تعاقدية\"},\n",
    "    {\"id\": 14, \"text\": \"التدابير الامنية التقنية والتنظيمية تشفير وصول حماية\"},\n",
    "    {\"id\": 15, \"text\": \"الابلاغ عن خرق البيانات للمركز خلال 72 ساعة والمتضرر 3 ايام\"},\n",
    "    {\"id\": 16, \"text\": \"مسؤول حماية البيانات DPO وبيانات التواصل معه\"},\n",
    "    {\"id\": 17, \"text\": \"التراخيص والتصاريح من مركز حماية البيانات المصري\"},\n",
    "    {\"id\": 18, \"text\": \"حماية بيانات الاطفال تحت 18 وموافقة ولي الامر\"},\n",
    "    {\"id\": 19, \"text\": \"سجلات انشطة المعالجة للرقابة والتفتيش\"},\n",
    "    {\"id\": 20, \"text\": \"وضوح اللغة العربية وسهولة الوصول للسياسة\"},\n",
    "    {\"id\": 21, \"text\": \"آليات التظلم والشكوى لمركز حماية البيانات المصري\"},\n",
    "]\n",
    "\n",
    "print(\"جاري بناء Dataset التدريب...\")\n",
    "print(\"=\" * 40)\n",
    "\n",
    "q_texts = [q[\"text\"] for q in questions_db]\n",
    "q_vectors = model.encode(q_texts)\n",
    "\n",
    "training_data = []\n",
    "\n",
    "print(\"استخراج امثلة الامتثال من القانون...\")\n",
    "for q, q_vec in zip(questions_db, q_vectors):\n",
    "    sims = cosine_similarity([q_vec], law_vectors)[0]\n",
    "    top_indices = np.argsort(sims)[-5:][::-1]\n",
    "    for idx in top_indices:\n",
    "        if sims[idx] > 0.35:\n",
    "            training_data.append({\n",
    "                \"text\": law_texts[idx],\n",
    "                \"question_id\": q[\"id\"],\n",
    "                \"label\": 1,\n",
    "                \"source\": \"law\",\n",
    "                \"similarity\": float(sims[idx])\n",
    "            })\n",
    "\n",
    "print(\"استخراج امثلة المخالفات من العقوبات...\")\n",
    "for q, q_vec in zip(questions_db, q_vectors):\n",
    "    sims = cosine_similarity([q_vec], penalty_vectors)[0]\n",
    "    top_indices = np.argsort(sims)[-5:][::-1]\n",
    "    for idx in top_indices:\n",
    "        if sims[idx] > 0.35:\n",
    "            training_data.append({\n",
    "                \"text\": penalty_texts[idx],\n",
    "                \"question_id\": q[\"id\"],\n",
    "                \"label\": 0,\n",
    "                \"source\": \"penalty\",\n",
    "                \"similarity\": float(sims[idx])\n",
    "            })\n",
    "\n",
    "df = pd.DataFrame(training_data).drop_duplicates(subset=[\"text\"])\n",
    "\n",
    "print(\"=\" * 40)\n",
    "print(f\"Dataset جاهز!\")\n",
    "print(f\"اجمالي الامثلة: {len(df)}\")\n",
    "print(f\"امتثال (1): {len(df[df.label == 1])}\")\n",
    "print(f\"مخالفة (0): {len(df[df.label == 0])}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "1c15886c-56aa-4b3d-8f56-3426c5e0487f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "جاري تجهيز بيانات التدريب...\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "9953aec8605c4a72abd55cf824a2901b",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Batches:   0%|          | 0/6 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "بيانات التدريب: 149\n",
      "بيانات الاختبار: 38\n",
      "\n",
      "جاري التدريب...\n",
      "\n",
      "نتائج التقييم:\n",
      "========================================\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "      مخالفة       0.62      0.80      0.70        20\n",
      "      امتثال       0.67      0.44      0.53        18\n",
      "\n",
      "    accuracy                           0.63        38\n",
      "   macro avg       0.64      0.62      0.61        38\n",
      "weighted avg       0.64      0.63      0.62        38\n",
      "\n",
      "تم حفظ النموذج!\n"
     ]
    }
   ],
   "source": [
    "from sklearn.neural_network import MLPClassifier\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.metrics import classification_report\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "import numpy as np\n",
    "import pickle\n",
    "\n",
    "print(\"جاري تجهيز بيانات التدريب...\")\n",
    "\n",
    "# تحويل نصوص الـ Dataset لـ vectors\n",
    "X = model.encode(df[\"text\"].tolist(), show_progress_bar=True)\n",
    "y = df[\"label\"].values\n",
    "\n",
    "# تطبيع البيانات\n",
    "scaler = StandardScaler()\n",
    "X_scaled = scaler.fit_transform(X)\n",
    "\n",
    "# تقسيم للتدريب والاختبار\n",
    "X_train, X_test, y_train, y_test = train_test_split(\n",
    "    X_scaled, y,\n",
    "    test_size=0.2,\n",
    "    random_state=42,\n",
    "    stratify=y\n",
    ")\n",
    "\n",
    "print(f\"بيانات التدريب: {len(X_train)}\")\n",
    "print(f\"بيانات الاختبار: {len(X_test)}\")\n",
    "\n",
    "# تدريب Neural Network\n",
    "print(\"\\nجاري التدريب...\")\n",
    "nn = MLPClassifier(\n",
    "    hidden_layer_sizes=(256, 128, 64),\n",
    "    activation='relu',\n",
    "    solver='adam',\n",
    "    max_iter=500,\n",
    "    random_state=42,\n",
    "    early_stopping=True,\n",
    "    validation_fraction=0.15\n",
    ")\n",
    "\n",
    "nn.fit(X_train, y_train)\n",
    "\n",
    "# تقييم النموذج\n",
    "y_pred = nn.predict(X_test)\n",
    "print(\"\\nنتائج التقييم:\")\n",
    "print(\"=\" * 40)\n",
    "print(classification_report(\n",
    "    y_test, y_pred,\n",
    "    target_names=[\"مخالفة\", \"امتثال\"]\n",
    "))\n",
    "\n",
    "# حفظ النموذج\n",
    "with open(\"nn_model.pkl\", \"wb\") as f:\n",
    "    pickle.dump(nn, f)\n",
    "with open(\"scaler.pkl\", \"wb\") as f:\n",
    "    pickle.dump(scaler, f)\n",
    "\n",
    "print(\"تم حفظ النموذج!\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "0348e777-c31f-4ff9-838d-f803b35bc035",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "السياسات المتاحة:\n",
      "1. google_privacy_policy_en.pdf\n",
      "2. Privacy policy Spotify .pdf\n",
      "3. privacy policy NAB.pdf\n",
      "\n",
      "الشات بوت جاهز!\n"
     ]
    }
   ],
   "source": [
    "import pickle\n",
    "import numpy as np\n",
    "from sklearn.metrics.pairwise import cosine_similarity\n",
    "\n",
    "with open(\"nn_model.pkl\", \"rb\") as f:\n",
    "    nn = pickle.load(f)\n",
    "with open(\"scaler.pkl\", \"rb\") as f:\n",
    "    scaler = pickle.load(f)\n",
    "\n",
    "def chatbot_answer(user_question, policy_name=None):\n",
    "    if policy_name is None:\n",
    "        policy_name = list(policies.keys())[0]\n",
    "\n",
    "    policy_texts_list = policies[policy_name]\n",
    "    policy_vecs = policy_vectors[policy_name]\n",
    "\n",
    "    q_vec = model.encode([user_question])\n",
    "\n",
    "    sims = cosine_similarity(q_vec, policy_vecs)[0]\n",
    "    top_idx = np.argsort(sims)[-3:][::-1]\n",
    "\n",
    "    law_sims = cosine_similarity(q_vec, law_vectors)[0]\n",
    "    law_idx = np.argmax(law_sims)\n",
    "    law_ref = law_texts[law_idx]\n",
    "\n",
    "    evidence_texts = [policy_texts_list[i] for i in top_idx]\n",
    "    evidence_vecs = model.encode(evidence_texts)\n",
    "    evidence_scaled = scaler.transform(evidence_vecs)\n",
    "    probabilities = nn.predict_proba(evidence_scaled)\n",
    "\n",
    "    avg_prob_compliant = np.mean(probabilities[:, 1])\n",
    "\n",
    "    if avg_prob_compliant > 0.6:\n",
    "        verdict = \"تلتزم\"\n",
    "        icon = \"OK\"\n",
    "    elif avg_prob_compliant > 0.4:\n",
    "        verdict = \"تلتزم جزئيا\"\n",
    "        icon = \"WARNING\"\n",
    "    else:\n",
    "        verdict = \"لا تلتزم\"\n",
    "        icon = \"VIOLATION\"\n",
    "\n",
    "    company = policy_name.replace(\".pdf\", \"\")\n",
    "    answer = (\n",
    "        \"\\n\" + \"=\"*50 + \"\\n\"\n",
    "        \"السؤال: \" + user_question + \"\\n\"\n",
    "        \"الشركة: \" + company + \"\\n\"\n",
    "        + \"=\"*50 + \"\\n\"\n",
    "        \"الحكم: \" + icon + \" الشركة \" + verdict + \"\\n\"\n",
    "        \"الثقة: \" + str(round(avg_prob_compliant*100)) + \"%\\n\\n\"\n",
    "        \"الدليل من السياسة:\\n\"\n",
    "        + evidence_texts[0][:200] + \"\\n\\n\"\n",
    "        \"المرجع القانوني:\\n\"\n",
    "        + law_ref[:200] + \"\\n\"\n",
    "        + \"=\"*50\n",
    "    )\n",
    "    return answer\n",
    "\n",
    "print(\"السياسات المتاحة:\")\n",
    "for i, name in enumerate(policies.keys()):\n",
    "    print(str(i+1) + \". \" + name)\n",
    "\n",
    "print(\"\\nالشات بوت جاهز!\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "16ea63ff-b997-42dd-8144-8a55c153d949",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "تم حفظ كل حاجة!\n"
     ]
    }
   ],
   "source": [
    "import pickle\n",
    "\n",
    "# حفظ كل حاجة محتاجينها\n",
    "with open(\"nn_model.pkl\", \"wb\") as f:\n",
    "    pickle.dump(nn, f)\n",
    "with open(\"scaler.pkl\", \"wb\") as f:\n",
    "    pickle.dump(scaler, f)\n",
    "with open(\"law_texts.pkl\", \"wb\") as f:\n",
    "    pickle.dump(law_texts, f)\n",
    "with open(\"law_vectors.pkl\", \"wb\") as f:\n",
    "    pickle.dump(law_vectors, f)\n",
    "with open(\"policies.pkl\", \"wb\") as f:\n",
    "    pickle.dump(policies, f)\n",
    "with open(\"policy_vectors.pkl\", \"wb\") as f:\n",
    "    pickle.dump(policy_vectors, f)\n",
    "\n",
    "print(\"تم حفظ كل حاجة!\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "1d688398-edba-4e0f-8976-a2f55b383b21",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "تم انشاء app.py!\n",
      "دلوقتي افتحي Terminal وشغلي:\n",
      "streamlit run app.py\n"
     ]
    }
   ],
   "source": [
    "app_code = '''\n",
    "import streamlit as st\n",
    "import pickle\n",
    "import numpy as np\n",
    "import pdfplumber\n",
    "import os\n",
    "from sklearn.metrics.pairwise import cosine_similarity\n",
    "from sentence_transformers import SentenceTransformer\n",
    "\n",
    "st.set_page_config(\n",
    "    page_title=\"محلل سياسات الخصوصية\",\n",
    "    page_icon=\"🔒\",\n",
    "    layout=\"wide\"\n",
    ")\n",
    "\n",
    "@st.cache_resource\n",
    "def load_model():\n",
    "    return SentenceTransformer(\n",
    "        \"paraphrase-multilingual-MiniLM-L12-v2\"\n",
    "    )\n",
    "\n",
    "@st.cache_resource\n",
    "def load_data():\n",
    "    with open(\"nn_model.pkl\", \"rb\") as f:\n",
    "        nn = pickle.load(f)\n",
    "    with open(\"scaler.pkl\", \"rb\") as f:\n",
    "        scaler = pickle.load(f)\n",
    "    with open(\"law_texts.pkl\", \"rb\") as f:\n",
    "        law_texts = pickle.load(f)\n",
    "    with open(\"law_vectors.pkl\", \"rb\") as f:\n",
    "        law_vectors = pickle.load(f)\n",
    "    with open(\"policies.pkl\", \"rb\") as f:\n",
    "        policies = pickle.load(f)\n",
    "    with open(\"policy_vectors.pkl\", \"rb\") as f:\n",
    "        policy_vectors = pickle.load(f)\n",
    "    return nn, scaler, law_texts, law_vectors, policies, policy_vectors\n",
    "\n",
    "model = load_model()\n",
    "nn, scaler, law_texts, law_vectors, policies, policy_vectors = load_data()\n",
    "\n",
    "def analyze_question(question, policy_name):\n",
    "    policy_texts_list = policies[policy_name]\n",
    "    policy_vecs = policy_vectors[policy_name]\n",
    "\n",
    "    q_vec = model.encode([question])\n",
    "    sims = cosine_similarity(q_vec, policy_vecs)[0]\n",
    "    top_idx = np.argsort(sims)[-3:][::-1]\n",
    "\n",
    "    law_sims = cosine_similarity(q_vec, law_vectors)[0]\n",
    "    law_idx = np.argmax(law_sims)\n",
    "    law_ref = law_texts[law_idx]\n",
    "\n",
    "    evidence_texts = [policy_texts_list[i] for i in top_idx]\n",
    "    evidence_vecs = model.encode(evidence_texts)\n",
    "    evidence_scaled = scaler.transform(evidence_vecs)\n",
    "    probabilities = nn.predict_proba(evidence_scaled)\n",
    "    avg_prob = np.mean(probabilities[:, 1])\n",
    "\n",
    "    if avg_prob > 0.6:\n",
    "        verdict = \"تلتزم بهذا المتطلب\"\n",
    "        color = \"green\"\n",
    "        icon = \"✅\"\n",
    "    elif avg_prob > 0.4:\n",
    "        verdict = \"تلتزم جزئياً\"\n",
    "        color = \"orange\"\n",
    "        icon = \"⚠️\"\n",
    "    else:\n",
    "        verdict = \"لا تلتزم\"\n",
    "        color = \"red\"\n",
    "        icon = \"❌\"\n",
    "\n",
    "    return {\n",
    "        \"verdict\": verdict,\n",
    "        \"icon\": icon,\n",
    "        \"color\": color,\n",
    "        \"confidence\": round(avg_prob * 100),\n",
    "        \"evidence\": evidence_texts[0][:300],\n",
    "        \"law_ref\": law_ref[:300]\n",
    "    }\n",
    "\n",
    "def load_new_policy(uploaded_file):\n",
    "    texts = []\n",
    "    with pdfplumber.open(uploaded_file) as pdf:\n",
    "        for page in pdf.pages:\n",
    "            text = page.extract_text()\n",
    "            if text:\n",
    "                paragraphs = [\n",
    "                    p.strip() for p in text.split(\"\\\\n\")\n",
    "                    if len(p.strip()) > 40\n",
    "                ]\n",
    "                texts.extend(paragraphs)\n",
    "    if texts:\n",
    "        vecs = model.encode(texts, show_progress_bar=False)\n",
    "        return texts, vecs\n",
    "    return None, None\n",
    "\n",
    "# ======================================\n",
    "# واجهة الشات بوت\n",
    "# ======================================\n",
    "st.title(\"🔒 محلل سياسات الخصوصية\")\n",
    "st.markdown(\"### وفق القانون المصري 151 لسنة 2020\")\n",
    "st.markdown(\"---\")\n",
    "\n",
    "col1, col2 = st.columns([1, 2])\n",
    "\n",
    "with col1:\n",
    "    st.markdown(\"### اختاري السياسة\")\n",
    "\n",
    "    # رفع سياسة جديدة\n",
    "    uploaded = st.file_uploader(\n",
    "        \"ارفعي سياسة خصوصية جديدة (PDF)\",\n",
    "        type=[\"pdf\"]\n",
    "    )\n",
    "\n",
    "    if uploaded:\n",
    "        with st.spinner(\"جاري تحليل السياسة...\"):\n",
    "            texts, vecs = load_new_policy(uploaded)\n",
    "            if texts:\n",
    "                policies[uploaded.name] = texts\n",
    "                policy_vectors[uploaded.name] = vecs\n",
    "                st.success(f\"تم تحميل: {uploaded.name}\")\n",
    "            else:\n",
    "                st.error(\"الملف فارغ او غير قابل للقراءة\")\n",
    "\n",
    "    # اختيار السياسة\n",
    "    selected_policy = st.selectbox(\n",
    "        \"اختاري من السياسات المتاحة:\",\n",
    "        list(policies.keys())\n",
    "    )\n",
    "\n",
    "    st.markdown(\"### اسئلة مقترحة\")\n",
    "    suggested = [\n",
    "        \"هل الشركة بتبيع بياناتي؟\",\n",
    "        \"هل يحق لي حذف بياناتي؟\",\n",
    "        \"هل الشركة بتشارك بياناتي مع اطراف تانية؟\",\n",
    "        \"هل الشركة بتحمي بياناتي من الاختراق؟\",\n",
    "        \"هل محتاج اوافق قبل جمع بياناتي؟\",\n",
    "        \"هل الشركة بتحتفظ ببياناتي لفترة طويلة؟\",\n",
    "    ]\n",
    "    for s in suggested:\n",
    "        if st.button(s, key=s):\n",
    "            st.session_state[\"question\"] = s\n",
    "\n",
    "with col2:\n",
    "    st.markdown(\"### اسأل عن السياسة\")\n",
    "\n",
    "    if \"messages\" not in st.session_state:\n",
    "        st.session_state.messages = []\n",
    "\n",
    "    # عرض المحادثة\n",
    "    for msg in st.session_state.messages:\n",
    "        with st.chat_message(msg[\"role\"]):\n",
    "            st.markdown(msg[\"content\"])\n",
    "\n",
    "    # سؤال من الاقتراحات\n",
    "    default_q = st.session_state.get(\"question\", \"\")\n",
    "\n",
    "    question = st.chat_input(\"اكتبي سؤالك هنا...\")\n",
    "\n",
    "    if question or default_q:\n",
    "        q = question or default_q\n",
    "        if default_q:\n",
    "            st.session_state[\"question\"] = \"\"\n",
    "\n",
    "        # عرض سؤال المستخدم\n",
    "        with st.chat_message(\"user\"):\n",
    "            st.markdown(q)\n",
    "        st.session_state.messages.append({\n",
    "            \"role\": \"user\",\n",
    "            \"content\": q\n",
    "        })\n",
    "\n",
    "        # تحليل وعرض الرد\n",
    "        with st.chat_message(\"assistant\"):\n",
    "            with st.spinner(\"جاري التحليل...\"):\n",
    "                result = analyze_question(q, selected_policy)\n",
    "\n",
    "            company = selected_policy.replace(\".pdf\", \"\")\n",
    "            response = f\"\"\"\n",
    "**الشركة:** {company}\n",
    "\n",
    "**الحكم:** {result[\"icon\"]} {result[\"verdict\"]}\n",
    "\n",
    "**نسبة الثقة:** {result[\"confidence\"]}%\n",
    "\n",
    "**الدليل من السياسة:**\n",
    "> {result[\"evidence\"]}\n",
    "\n",
    "**المرجع القانوني:**\n",
    "> {result[\"law_ref\"]}\n",
    "\"\"\"\n",
    "            st.markdown(response)\n",
    "            st.session_state.messages.append({\n",
    "                \"role\": \"assistant\",\n",
    "                \"content\": response\n",
    "            })\n",
    "'''\n",
    "\n",
    "with open(\"app.py\", \"w\", encoding=\"utf-8\") as f:\n",
    "    f.write(app_code)\n",
    "\n",
    "print(\"تم انشاء app.py!\")\n",
    "print(\"دلوقتي افتحي Terminal وشغلي:\")\n",
    "print(\"streamlit run app.py\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "9860079f-2b9e-4b85-b7b1-fac9c5326802",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "تم انشاء requirements.txt\n",
      "دلوقتي قوليلي:\n",
      "1. عندك حساب GitHub؟\n",
      "2. عندك حساب Streamlit Cloud؟\n"
     ]
    }
   ],
   "source": [
    "# عمل requirements.txt\n",
    "requirements = \"\"\"sentence-transformers\n",
    "pdfplumber\n",
    "torch\n",
    "scikit-learn\n",
    "pandas\n",
    "numpy\n",
    "streamlit\"\"\"\n",
    "\n",
    "with open(\"requirements.txt\", \"w\") as f:\n",
    "    f.write(requirements)\n",
    "\n",
    "print(\"تم انشاء requirements.txt\")\n",
    "print(\"دلوقتي قوليلي:\")\n",
    "print(\"1. عندك حساب GitHub؟\")\n",
    "print(\"2. عندك حساب Streamlit Cloud؟\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "125dc64b-819c-4c73-9234-f4ba5fe4334c",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "تم انشاء requirements.txt\n"
     ]
    }
   ],
   "source": [
    "# عمل requirements.txt\n",
    "requirements = \"\"\"sentence-transformers\n",
    "pdfplumber\n",
    "torch\n",
    "scikit-learn\n",
    "pandas\n",
    "numpy\n",
    "streamlit\"\"\"\n",
    "\n",
    "with open(\"requirements.txt\", \"w\", encoding=\"utf-8\") as f:\n",
    "    f.write(requirements)\n",
    "\n",
    "print(\"تم انشاء requirements.txt\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "12a6a988-362c-448a-906b-64328e6e8d12",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
