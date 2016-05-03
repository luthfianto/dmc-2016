# dmc-2016

- `python make_train_dataset_only.py` atau ekstrak fiturmu sendiri
- load di ipython notebook
- ganti/pilih fiturmu sendiri
- split dengan rasio=0.15 (85% + 15%) agar sama dengan jumlah data testing. diubah juga boleh, tapi bilang ya
- prediksi deh. terserah mau pakai model regressor apa
- turunkan error masing-masing
- kalau errornya masih jelek, jangan berkecil hati. siapa tau pas di-ensemble malah bikin bagus, karena ensemble yang baik justru yang hasil/metodenya berbeda

### Yang harus dikirim menjelang hari akhir

- prediksi 15% data training  tanpa dibulatkan ke integer, dari model yang dilatih dengan 85% data training
- pastikan 15% data tersebut tidak mengalami dropna. (data 85%-nya terserah deh kalau mau di-dropna asal yakin) 
- prediksi data testing, dari model yang dilatih dengan 100% data training
- pastikan data testing tidak mengalami dropna. jumlahnya harus 341099.
- pemilihan fitur bebas. hindari *leaking*. kalau misal hasilnya terlalu bagus, kabarkan ke teman-teman dan sambil di-cek lagi mengapa 
