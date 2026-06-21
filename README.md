# Saudi Companies Directory

## المحتويات
- `app.py` — التطبيق الرئيسي (Streamlit): فيه 4 تبويبات
  - Companies List (الفلترة)
  - Analytics (رسومات)
  - Lead Scoring (تقييم الشركات بالذكاء الاصطناعي)
  - Campaign (توليد إيميلات للشركات المختارة) ← **الجديد**
- `scorer.py` — منطق تقييم الشركات (يستخدم OpenAI + اختياري Anthropic للبحث)
- `campaign_agent.py` — منطق توليد الإيميلات (يستخدم OpenAI)
- `companies.json` — قاعدة بيانات الشركات
- `requirements.txt` — المكتبات المطلوبة
- `.env.example` — مثال لملف المفاتيح

## خطوات التشغيل

### 1) تثبيت المكتبات
```bash
pip install -r requirements.txt
```

### 2) إضافة مفاتيح API
انسخي ملف `.env.example` وسميه `.env`، وضعي مفاتيحك الحقيقية بداخله:
```bash
cp .env.example .env
```
بعدين افتحي `.env` واستبدلي:
```
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```
بمفاتيحك الحقيقية. لازم على الأقل `OPENAI_API_KEY` يكون موجود وصحيح.
`ANTHROPIC_API_KEY` يحتاج فقط لو رحتي تستخدمين "🤖 Agent Mode" في تبويب Lead Scoring.

### 3) تشغيل التطبيق
```bash
streamlit run app.py
```
بيفتح في المتصفح على `localhost:8501`.

## كيف تستخدمين تبويب Campaign الجديد

1. روحي تبويب **🎯 Lead Scoring** أولاً واضغطي "Score Companies" عشان يطلع لك تقييم للشركات.
2. روحي تبويب **📧 Campaign**.
3. علّمي (✓) على الشركات اللي تبين تبعتين لها إيميل.
4. اضغطي **"Generate Emails for X Selected Companies"**.
5. الإيميلات تظهر في الصفحة، وفي الأسفل زر لتصدير الكل كملف CSV.

## ملاحظة أمان مهمة
لا ترفعي ملف `.env` على GitHub أبداً (فيه مفاتيحك الحقيقية). الملف `.gitignore` المرفق يتجاهله تلقائياً.
