# dmc-2016

- `python make_train_dataset_only.py` atau ekstrak fiturmu sendiri
- load di ipython notebook
- ganti/pilih fiturmu sendiri
- split dengan rasio=0.15 (85% + 15%) agar sama dengan jumlah data testing. diubah juga boleh, tapi bilang ya
- prediksi deh. terserah mau pakai model regressor apa
- turunkan error masing-masing
- kalau errornya masih jelek, jangan berkecil hati. siapa tau pas di-ensemble malah bikin bagus, karena ensemble yang baik justru yang hasil/metodenya berbeda
- cek dan koreksi apakah predicted_returnQuantity > quantity

### Trivia

- Confusion matrix Linear Regression:
  - yg ketebak `0`: 117083, yg ditebak 1 (pdhl `0`): 50579
  - yg ditebak `0` (padahal `1`): 31544 yg ketebak 1 :148974
  - kan nyaris 50:50 yg `0`
- Confusion matrix Lasso (L1) kaya kebalikannya LinearRegression/Ridge (L2) gitu
  - PolynomialFeatures atau *feature selection* yang berbeda sangat [berpengaruh](https://github.com/rilut/dmc-2016/blob/master/notebook%2FCoba%20PolynomialFeatures.ipynb) kalau mau pake Lasso
    - dengan itu, error dari 136040 jadi 96384



### Yang harus dikirim menjelang hari akhir

- prediksi 15% data training  tanpa dibulatkan ke integer, dari model yang dilatih dengan 85% data training
- pastikan 15% data tersebut tidak mengalami dropna. (data 85%-nya terserah deh kalau mau di-dropna asal yakin) 
- prediksi data testing, dari model yang dilatih dengan 100% data training
- pastikan data testing tidak mengalami dropna. jumlahnya harus 341099.
- pemilihan fitur bebas. hindari *leaking*. kalau misal hasilnya terlalu bagus, kabarkan ke teman-teman dan sambil di-cek lagi mengapa 
