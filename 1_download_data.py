# 1_download_data.py
import os, requests

os.makedirs("nhanes_data", exist_ok=True)

BASE = {
    "1999-2000": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/1999/DataFiles/",
    "2001-2002": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2001/DataFiles/",
    "2003-2004": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2003/DataFiles/",
    "2005-2006": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2005/DataFiles/",
    "2007-2008": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2007/DataFiles/",
    "2009-2010": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2009/DataFiles/",
    "2011-2012": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2011/DataFiles/",
    "2013-2014": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/",
    "2015-2016": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/",
    "2017-2018": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/"
}

# File names per cycle (best-effort based on NHANES conventions)
FILES = {
    "1999-2000": {"DEMO":"DEMO.xpt","OHX":"OHXDENT.xpt","CRP":"LAB11.xpt","CBC":"LAB25.xpt",
                  "SMQ":"SMQ.xpt","ALQ":"ALQ.xpt","BMX":"BMX.xpt","DIQ":"DIQ.xpt","PIR":"DEMO.xpt",
                  "HGM":"LAB06HM.xpt"},
    "2001-2002": {"DEMO":"DEMO_B.xpt","OHX":"OHXDEN_B.xpt","CRP":"L11_B.xpt","CBC":"LAB25_B.xpt",
                  "SMQ":"SMQ_B.xpt","ALQ":"ALQ_B.xpt","BMX":"BMX_B.xpt","DIQ":"DIQ_B.xpt","PIR":"DEMO_B.xpt",
                  "HGM":"L06_2_B.xpt"},
    "2003-2004": {"DEMO":"DEMO_C.xpt","OHX":"OHXDEN_C.xpt","CRP":"L11_C.xpt","CBC":"LAB25_C.xpt",
                  "SMQ":"SMQ_C.xpt","ALQ":"ALQ_C.xpt","BMX":"BMX_C.xpt","DIQ":"DIQ_C.xpt","PIR":"DEMO_C.xpt",
                  "HGM":"L06BMT_C.xpt"},
    "2005-2006": {"DEMO":"DEMO_D.xpt","OHX":"OHXDEN_D.xpt","CRP":"CRP_D.xpt","CBC":"CBC_D.xpt",
                  "SMQ":"SMQ_D.xpt","ALQ":"ALQ_D.xpt","BMX":"BMX_D.xpt","DIQ":"DIQ_D.xpt","PIR":"DEMO_D.xpt",
                  "HGM":"PBCD_D.xpt"},
    "2007-2008": {"DEMO":"DEMO_E.xpt","OHX":"OHXDEN_E.xpt","CRP":"CRP_E.xpt","CBC":"CBC_E.xpt",
                  "SMQ":"SMQ_E.xpt","ALQ":"ALQ_E.xpt","BMX":"BMX_E.xpt","DIQ":"DIQ_E.xpt","PIR":"DEMO_E.xpt",
                  "HGM":"PBCD_E.xpt"},
    "2009-2010": {"DEMO":"DEMO_F.xpt","OHX":"OHXDEN_F.xpt","CRP":"CRP_F.xpt","CBC":"CBC_F.xpt",
                  "SMQ":"SMQ_F.xpt","ALQ":"ALQ_F.xpt","BMX":"BMX_F.xpt","DIQ":"DIQ_F.xpt","PIR":"DEMO_F.xpt",
                  "HGM":"PBCD_F.xpt"},
    "2011-2012": {"DEMO":"DEMO_G.xpt","OHX":"OHXDEN_G.xpt","CRP":"CRP_G.xpt","CBC":"CBC_G.xpt",
                  "SMQ":"SMQ_G.xpt","ALQ":"ALQ_G.xpt","BMX":"BMX_G.xpt","DIQ":"DIQ_G.xpt","PIR":"DEMO_G.xpt",
                  "HGM":"PBCD_G.xpt"},
    "2013-2014": {"DEMO":"DEMO_H.xpt","OHX":"OHXDEN_H.xpt","CRP":"CRP_H.xpt","CBC":"CBC_H.xpt",
                  "SMQ":"SMQ_H.xpt","ALQ":"ALQ_H.xpt","BMX":"BMX_H.xpt","DIQ":"DIQ_H.xpt","PIR":"DEMO_H.xpt",
                  "HGM":"PBCD_H.xpt"},
    "2015-2016": {"DEMO":"DEMO_I.xpt","OHX":"OHXDEN_I.xpt","CRP":"CRP_I.xpt","CBC":"CBC_I.xpt",
                  "SMQ":"SMQ_I.xpt","ALQ":"ALQ_I.xpt","BMX":"BMX_I.xpt","DIQ":"DIQ_I.xpt","PIR":"DEMO_I.xpt",
                  "HGM":"PBCD_I.xpt"},
    "2017-2018": {"DEMO":"DEMO_J.xpt","OHX":"OHXDEN_J.xpt","CRP":"HSCRP_J.xpt","CBC":"CBC_J.xpt",
                  "SMQ":"SMQ_J.xpt","ALQ":"ALQ_J.xpt","BMX":"BMX_J.xpt","DIQ":"DIQ_J.xpt","PIR":"DEMO_J.xpt",
                  "HGM":"PBCD_J.xpt"}
}

for cycle, base in BASE.items():
    print(f"\\n📦 Processing {cycle}")
    for label, fname in FILES[cycle].items():
        url = base + fname
        path = os.path.join("nhanes_data", fname)
        if os.path.exists(path) and os.path.getsize(path) > 0:
            print(f"⏭️  {fname} exists, skipping.")
            continue
        try:
            r = requests.get(url, timeout=60)
            if r.status_code == 200 and r.content:
                with open(path, "wb") as f:
                    f.write(r.content)
                print(f"✅ {label}: {fname}")
            else:
                print(f"⚠️ Missing or unavailable: {fname} ({r.status_code})")
        except Exception as e:
            print(f"❌ Error downloading {fname}: {e}")
print("Done.")
