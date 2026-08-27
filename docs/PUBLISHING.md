# GitHub Yayımlama Kontrol Listesi

1. `rg` veya bir secret scanner ile çalışma ağacını ve Git geçmişini tarayın.
2. `data/`, `vault/`, `logs/`, `.venv/` ve model dosyalarının izlenmediğini doğrulayın.
3. Gerçek ad, e-posta, mutlak kullanıcı yolu ve özel belge olmadığını kontrol edin.
4. `python -m unittest discover -s tests` çalıştırın.
5. Lisansı bilinçli olarak seçin; seçim yapılmadıysa lisans eklemeyin.
6. Yeni boş GitHub deposunu README, lisans veya `.gitignore` ile başlatmayın.
7. Temiz `main` dalını gönderin; eski yerel projenin Git geçmişini eklemeyin.
8. GitHub'da Dependabot, secret scanning, push protection, code scanning ve private
   vulnerability reporting seçeneklerini açın.
9. Web arayüzünde dosya listesini ve ilk commit yazar e-postasını son kez kontrol edin.
