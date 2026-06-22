# Saudi Companies Directory

## المحتويات
- `app.py` — التطبيق الرئيسي (Streamlit): فيه 4 
  - Companies List (الفلترة)
  - Analytics (رسومات)
  - Lead Scoring (تقييم الشركات بالذكاء الاصطناعي)
  - Campaign (توليد إيميلات للشركات المختارة) ← **الجديد**
- `scorer.py` — منطق تقييم الشركات (يستخدم OpenAI + اختياري Anthropic للبحث)
- `campaign_agent.py` — منطق توليد الإيميلات (يستخدم OpenAI)
- `companies.json` — قاعدة بيانات الشركات
- `requirements.txt` — المكتبات المطلوبة
- `.env.example` — مثال لملف المفاتيح

streamlit run app.py
