# Dapatkan probabilitas kumulatif terakhir
df[[kolom_kolom_prob]].drop_duplicates(cols = 'customerID', keep='last').to_dict()
