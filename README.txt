CROPWISE - PREMIUM BOTANICAL DASHBOARD

This package replaces the previous frontend design and includes a starter Flask app.py.

FILES
- app.py
- templates/base.html
- templates/dashboard.html
- templates/coming_soon.html
- static/css/style.css
- static/js/script.js

INSTALL / USE
1. Make a backup of your current templates and static folders if you want to preserve the old design.
2. Copy app.py into D:\croprecommend.
3. Merge/replace the templates and static folders with the ones in this package.
4. Keep your existing dataset, model, reports, data_cleaning.py, data_analysis.py and model_training.py files.
5. Run from D:\croprecommend:
   "D:\PYTHON\PYTHON_312\python.exe" app.py
6. Open http://127.0.0.1:5000/

IMPORTANT
The recommendation, model and about routes currently use a placeholder page. The trained model has not yet been connected to this frontend. We will implement that separately, using your existing saved model file and exact feature order:
N, P, K, temperature, humidity, ph, rainfall.

The dashboard displays the metrics reported in your training output: 2,200 records, 22 crop categories, 7 features, 99.55% test accuracy, and 99.49% mean cross-validation accuracy. These are dataset/model evaluation figures, not a guarantee of real-world crop suitability.

No database is used. Google Fonts are loaded from the internet; system fallback fonts are provided if unavailable.
