from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

print("Sedang memuat model ML 46 kategori... Tunggu sebentar...")

# ==========================================
# 1. LOAD FILE PENDUKUNG (46 CATEGORIES MODEL)
# ==========================================
try:
    model = joblib.load('model_rf_46cat.pkl')
    scaler = joblib.load('scaler_46cat.pkl')
    encoders = joblib.load('encoders_46cat.pkl')
    lookups = joblib.load('lookup_dicts_46cat.pkl')
    feature_cols = joblib.load('feature_columns_46cat.pkl')
    print("✅ Model 46 kategori berhasil di-load!")
except Exception as e:
    print(f"❌ ERROR: Gagal load file. Error: {e}")

# ==========================================
# 2. FUNGSI EKSTRAKSI KATEGORI (46 CATEGORIES)
# ==========================================

def extract_category_detailed(job):
    """
    Ekstraksi 46 kategori detail - LANGSUNG untuk model (no mapping needed).
    """
    job = str(job).lower()
    
    # TECH & IT (13 categories)
    if any(w in job for w in ['backend', 'back end', 'back-end']):
        return 'Backend Development'
    elif any(w in job for w in ['frontend', 'front end', 'front-end', 'react', 'vue', 'angular']):
        return 'Frontend Development'
    elif any(w in job for w in ['full stack', 'fullstack', 'full-stack']):
        return 'Full Stack Development'
    elif any(w in job for w in ['mobile developer', 'android developer', 'ios developer', 'flutter']):
        return 'Mobile Development'
    elif any(w in job for w in ['data analyst', 'data scientist', 'business intelligence', 'bi analyst']):
        return 'Data Science & Analytics'
    elif any(w in job for w in ['data engineer', 'etl', 'big data']):
        return 'Data Engineering'
    elif any(w in job for w in ['machine learning', 'ml engineer', 'ai engineer', 'artificial intelligence']):
        return 'Machine Learning & AI'
    elif any(w in job for w in ['devops', 'cloud engineer', 'aws', 'azure', 'kubernetes', 'docker']):
        return 'DevOps & Cloud Engineering'
    elif any(w in job for w in ['system admin', 'sysadmin', 'network engineer', 'network admin']):
        return 'System Administration'
    elif any(w in job for w in ['it support', 'technical support', 'helpdesk', 'help desk']):
        return 'IT Support & Helpdesk'
    elif any(w in job for w in ['quality assurance', 'qa engineer', 'tester', 'test engineer']):
        return 'Quality Assurance & Testing'
    elif any(w in job for w in ['security', 'cybersecurity', 'information security', 'infosec']):
        return 'Cybersecurity'
    elif any(w in job for w in ['developer', 'programmer', 'software engineer']):
        return 'Software Development'
    
    # BUSINESS (7 categories)
    elif any(w in job for w in ['sales executive', 'sales officer', 'sales manager', 'account manager', 'relationship officer']):
        return 'Sales & Account Management'
    elif any(w in job for w in ['business development', 'bbd', 'business partner']):
        return 'Business Development'
    elif any(w in job for w in ['store crew', 'store manager', 'cashier', 'kasir', 'retail', 'merchandiser']):
        return 'Retail & Store Management'
    elif any(w in job for w in ['brand manager', 'marketing manager', 'marketing executive', 'trade marketing']):
        return 'Marketing & Brand Management'
    elif any(w in job for w in ['digital marketing', 'social media', 'seo', 'sem', 'performance marketing']):
        return 'Digital Marketing & Social Media'
    elif any(w in job for w in ['content creator', 'content writer', 'copywriter', 'video editor', 'videographer']):
        return 'Content Creation & Media'
    elif any(w in job for w in ['product manager', 'product owner', 'product specialist']):
        return 'Product Management'
    elif 'sales' in job or 'penjualan' in job:
        return 'Sales & Account Management'
    elif 'marketing' in job:
        return 'Marketing & Brand Management'
    
    # FINANCE & ACCOUNTING (3 categories)
    elif any(w in job for w in ['finance manager', 'finance officer', 'financial analyst', 'treasury']):
        return 'Finance & Treasury'
    elif any(w in job for w in ['accounting', 'accountant', 'akuntan', 'audit', 'auditor']):
        return 'Accounting & Auditing'
    elif any(w in job for w in ['tax', 'pajak']):
        return 'Tax & Compliance'
    
    # OPERATIONS (6 categories)
    elif any(w in job for w in ['operation manager', 'operational manager', 'operation supervisor']):
        return 'Operations Management'
    elif any(w in job for w in ['project manager', 'project officer', 'project coordinator']):
        return 'Project Management'
    elif any(w in job for w in ['purchasing', 'procurement', 'buyer']):
        return 'Procurement & Purchasing'
    elif any(w in job for w in ['supply chain', 'logistics', 'logistik', 'warehouse', 'distribution']):
        return 'Supply Chain & Logistics'
    elif any(w in job for w in ['production', 'produksi', 'manufacturing', 'operator produksi']):
        return 'Production & Manufacturing'
    elif any(w in job for w in ['quality control', 'qc', 'qc inspector']):
        return 'Quality Control & Assurance'
    elif 'operation' in job or 'operational' in job:
        return 'Operations Management'
    
    # HR & ADMIN (3 categories)
    elif any(w in job for w in ['human resource', 'hr manager', 'hr officer', 'hrga', 'hrd', 'human capital']):
        return 'Human Resources & GA'
    elif any(w in job for w in ['recruitment', 'recruiter', 'talent acquisition']):
        return 'Recruitment & Talent'
    elif any(w in job for w in ['admin', 'administration', 'administrasi', 'secretary', 'sekretaris']):
        return 'Administration & Secretary'
    
    # CUSTOMER SERVICE (2 categories)
    elif any(w in job for w in ['customer service', 'customer care', 'customer support', 'call center']):
        return 'Customer Service'
    elif any(w in job for w in ['customer success', 'client success']):
        return 'Customer Success'
    
    # DESIGN (2 categories)
    elif any(w in job for w in ['designer', 'design', 'ui designer', 'ux designer', 'ui/ux', 'graphic designer']):
        return 'Design & UI/UX'
    elif any(w in job for w in ['creative', 'art director', 'illustrator']):
        return 'Creative & Content Production'
    
    # ENGINEERING & TECHNICAL (2 categories)
    elif any(w in job for w in ['engineer', 'engineering', 'mechanical', 'electrical', 'civil', 'teknisi']):
        return 'Engineering & Technical'
    elif any(w in job for w in ['maintenance', 'facility']):
        return 'Maintenance & Facilities'
    
    # OTHERS (9 categories)
    elif any(w in job for w in ['dokter', 'doctor', 'nurse', 'perawat', 'apoteker', 'pharmacist', 'medical']):
        return 'Medical & Healthcare'
    elif any(w in job for w in ['teacher', 'guru', 'trainer', 'instructor', 'tutor']):
        return 'Education & Training'
    elif any(w in job for w in ['legal', 'lawyer', 'compliance', 'regulatory']):
        return 'Legal & Compliance'
    elif any(w in job for w in ['consultant', 'konsultan', 'advisory', 'advisor']):
        return 'Consulting & Advisory'
    elif any(w in job for w in ['hotel', 'hospitality', 'restaurant', 'chef', 'barista', 'f&b']):
        return 'Hospitality & F&B'
    elif any(w in job for w in ['property', 'real estate', 'site manager']):
        return 'Property & Real Estate'
    elif any(w in job for w in ['collection', 'collector', 'credit']):
        return 'Collection & Credit'
    elif any(w in job for w in ['branch manager', 'area manager', 'general manager', 'management trainee']):
        return 'General Management'
    
    else:
        return 'General Business'


def extract_level(job):
    """Ekstraksi level jabatan - return numeric 1-7"""
    job = str(job).lower()
    
    if any(w in job for w in ['ceo', 'cto', 'cfo', 'coo', 'cmo', 'president', 'chief', 'direktur utama']):
        return 7
    elif any(w in job for w in ['vp', 'vice president']):
        return 6
    elif any(w in job for w in ['director', 'direktur', 'head of', 'head ']):
        return 5
    elif any(w in job for w in ['manager', 'mgr', 'manajer', 'supervisor', 'spv']):
        return 4
    elif any(w in job for w in ['senior', 'sr.', 'sr ', 'principal', 'lead', 'expert']):
        return 3
    elif any(w in job for w in ['junior', 'jr.', 'jr ', 'trainee', 'intern', 'magang', 'assistant', 'asisten']):
        return 1
    else:
        return 2


def extract_tier(loc):
    """Ekstraksi tier lokasi"""
    loc = str(loc).lower()
    
    if any(w in loc for w in ['jakarta pusat', 'jakarta selatan', 'jakarta barat']):
        return 'Tier_1A_Jakarta_Core'
    elif any(w in loc for w in ['jakarta', 'tangerang selatan']):
        return 'Tier_1B_Jakarta_Area'
    elif any(w in loc for w in ['bekasi', 'depok', 'bogor', 'tangerang']):
        return 'Tier_1C_Jabodetabek'
    elif any(w in loc for w in ['surabaya', 'bandung', 'medan']):
        return 'Tier_2A_Major'
    elif any(w in loc for w in ['semarang', 'palembang', 'makassar', 'bali', 'denpasar', 'yogyakarta']):
        return 'Tier_2B_Big'
    else:
        return 'Tier_3_Other'


def get_level_name(level_num):
    """Convert level number ke nama"""
    mapping = {1: 'Junior', 2: 'Mid-Level', 3: 'Senior', 4: 'Manager', 
               5: 'Director', 6: 'Vice President', 7: 'C-Level'}
    return mapping.get(level_num, 'Unknown')


# ==========================================
# 3. ENDPOINT API
# ==========================================

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        input_posisi = data.get('posisi', '')
        input_lokasi = data.get('lokasi', '')
        input_perusahaan = data.get('perusahaan', 'Unknown Company')
        
        if not input_posisi or not input_lokasi:
            return jsonify({'status': 'error', 'message': 'Posisi dan Lokasi harus diisi'}), 400

        print(f"📥 Input: {input_posisi} | {input_lokasi}")

        # Buat DataFrame
        df = pd.DataFrame([{
            'Judul Pekerjaan': input_posisi,
            'Lokasi': input_lokasi,
            'Perusahaan': input_perusahaan
        }])

        # FEATURE ENGINEERING
        df['Level_Encoded'] = df['Judul Pekerjaan'].apply(extract_level)
        
        # Kategori (langsung, tidak perlu mapping!)
        category = extract_category_detailed(input_posisi)
        print(f"📂 Kategori: {category}")
        
        try:
            df['Category_Encoded'] = encoders['category'].transform([category])[0]
        except:
            df['Category_Encoded'] = 0
            print(f"⚠️ Category '{category}' tidak dikenali")
        
        tier_str = extract_tier(input_lokasi)
        try:
            df['Tier_Encoded'] = encoders['tier'].transform([tier_str])[0]
        except:
            df['Tier_Encoded'] = 0

        # Statistical Features
        global_mean = lookups['mean_salary_overall']
        
        df['Lokasi_Mean_Salary'] = lookups['lokasi_mean'].get(input_lokasi, global_mean)
        df['Judul_Mean_Salary'] = lookups['judul_mean'].get(input_posisi, global_mean)
        df['Perusahaan_Mean_Salary'] = lookups['perusahaan_mean'].get(input_perusahaan, global_mean)
        df['Category_Mean_Salary'] = lookups['category_mean'].get(category, global_mean)
        
        df['Lokasi_Median_Salary'] = lookups['lokasi_median'].get(input_lokasi, global_mean)
        df['Judul_Median_Salary'] = lookups['judul_median'].get(input_posisi, global_mean)
        
        df['Lokasi_Frequency'] = lookups['lokasi_freq'].get(input_lokasi, 0.001)
        df['Judul_Frequency'] = lookups['judul_freq'].get(input_posisi, 0.001)
        df['Perusahaan_Frequency'] = lookups['perusahaan_freq'].get(input_perusahaan, 0.001)

        # Text Features
        df['Panjang_Judul'] = len(input_posisi)
        df['Jumlah_Kata_Judul'] = len(input_posisi.split())
        df['Panjang_Lokasi'] = len(input_lokasi)
        df['Panjang_Perusahaan'] = len(input_perusahaan)
        
        # Boolean Features
        pos_lower = input_posisi.lower()
        df['Has_Number'] = 1 if any(c.isdigit() for c in input_posisi) else 0
        df['Has_Bracket'] = 1 if '(' in input_posisi or ')' in input_posisi else 0
        
        df['Is_Manager'] = 1 if any(w in pos_lower for w in ['manager', 'manajer']) else 0
        df['Is_Senior'] = 1 if any(w in pos_lower for w in ['senior', 'sr']) else 0
        df['Is_Junior'] = 1 if any(w in pos_lower for w in ['junior', 'jr']) else 0
        df['Is_Lead'] = 1 if any(w in pos_lower for w in ['lead', 'head']) else 0
        df['Is_Assistant'] = 1 if any(w in pos_lower for w in ['assistant', 'asisten']) else 0
        df['Is_Specialist'] = 1 if any(w in pos_lower for w in ['specialist', 'spesialis']) else 0
        
        df['Is_Tech'] = 1 if any(x in pos_lower for x in ['developer', 'engineer', 'programmer', 'it', 'tech', 'data', 'software']) else 0
        df['Is_Sales'] = 1 if any(w in pos_lower for w in ['sales', 'penjualan']) else 0
        df['Is_Marketing'] = 1 if 'marketing' in pos_lower else 0
        df['Is_Finance'] = 1 if any(w in pos_lower for w in ['finance', 'accounting', 'akuntan']) else 0

        # Interaction Features
        df['Level_x_Tier'] = df['Level_Encoded'] * df['Tier_Encoded']
        df['Level_x_Category'] = df['Level_Encoded'] * df['Category_Encoded']
        df['Category_x_Tier'] = df['Category_Encoded'] * df['Tier_Encoded']
        
        df['Level_Squared'] = df['Level_Encoded'] ** 2
        df['Panjang_Judul_Squared'] = df['Panjang_Judul'] ** 2
        
        # Ratio Features
        df['Salary_vs_Mean_Ratio'] = df['Lokasi_Mean_Salary'] / global_mean
        df['Judul_Salary_vs_Mean'] = df['Judul_Mean_Salary'] / global_mean

        # Susun kolom
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0
        
        df_final = df[feature_cols]
        
        # Scaling & Prediction
        X_scaled = scaler.transform(df_final)
        pred_log = model.predict(X_scaled)[0]
        pred_rupiah = np.expm1(pred_log)
        hasil_akhir = round(pred_rupiah, -4)

        print(f"💰 Prediksi: Rp {hasil_akhir:,.0f}")

        return jsonify({
            'status': 'success',
            'estimasi_gaji': float(hasil_akhir),
            'posisi_diterima': input_posisi,
            'lokasi_diterima': input_lokasi,
            'kategori_pekerjaan': category,
            'tier_lokasi': tier_str,
            'level_jabatan': int(df['Level_Encoded'].iloc[0]),
            'level_jabatan_nama': get_level_name(int(df['Level_Encoded'].iloc[0])),
            'model_version': '46_categories'
        })

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'message': 'API berjalan dengan baik',
        'model_version': '46_categories',
        'model_loaded': model is not None
    })


@app.route('/categories', methods=['GET'])
def get_categories():
    """List semua 46 kategori"""
    categories = [
        # Tech & IT
        'Backend Development', 'Frontend Development', 'Full Stack Development',
        'Mobile Development', 'Data Science & Analytics', 'Data Engineering',
        'Machine Learning & AI', 'DevOps & Cloud Engineering', 'System Administration',
        'IT Support & Helpdesk', 'Quality Assurance & Testing', 'Cybersecurity',
        'Software Development',
        
        # Business
        'Sales & Account Management', 'Business Development', 'Retail & Store Management',
        'Marketing & Brand Management', 'Digital Marketing & Social Media',
        'Content Creation & Media', 'Product Management',
        
        # Finance
        'Finance & Treasury', 'Accounting & Auditing', 'Tax & Compliance',
        
        # Operations
        'Operations Management', 'Project Management', 'Procurement & Purchasing',
        'Supply Chain & Logistics', 'Production & Manufacturing',
        'Quality Control & Assurance',
        
        # HR & Admin
        'Human Resources & GA', 'Recruitment & Talent', 'Administration & Secretary',
        
        # Customer Service
        'Customer Service', 'Customer Success',
        
        # Design
        'Design & UI/UX', 'Creative & Content Production',
        
        # Engineering
        'Engineering & Technical', 'Maintenance & Facilities',
        
        # Others
        'Medical & Healthcare', 'Education & Training', 'Legal & Compliance',
        'Consulting & Advisory', 'Hospitality & F&B', 'Property & Real Estate',
        'Collection & Credit', 'General Management', 'General Business'
    ]
    
    return jsonify({
        'status': 'success',
        'total_categories': len(categories),
        'categories': sorted(categories),
        'model_version': '46_categories'
    })


@app.route('/info', methods=['GET'])
def model_info():
    return jsonify({
        'status': 'success',
        'model': 'Random Forest Regressor',
        'model_version': '46_categories',
        'category_system': 'Direct prediction with 46 categories (no mapping)',
        'features_count': len(feature_cols),
        'expected_metrics': {
            'r2_score': '95-96%',
            'improvement': '+1-2% from 16-category model'
        }
    })


if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 FLASK API - PREDIKSI GAJI (46 CATEGORIES MODEL)")
    print("="*70)
    print(f"📊 Model: Random Forest (Expected R² = 95-96%)")
    print(f"📂 Kategori: 46 kategori (no mapping)")
    print(f"🎯 Direct prediction")
    print("="*70 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
