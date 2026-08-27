# ZESTOLES Güvenlik Modeli

## Güven sınırları

ZESTOLES aynı Windows kullanıcısı altında çalışan kişisel bir asistandır. HUD ve
WebSocket sunucusu yalnız loopback'e bağlanır. POST istekleri, servis başladığında
üretilip yalnız sunulan HTML'e yerleştirilen geçici belirteci taşımak zorundadır.
Bu mekanizma uzaktan kimlik doğrulama değildir; aynı Windows hesabı altında çalışan
kötü amaçlı bir süreç zaten kullanıcının dosyalarına ve loopback trafiğine erişebilir.

## Yetkili araçlar

Salt okuma araçları doğrudan çalışabilir. Dosya yazma/taşıma, pano yazma, uygulama
açma/kapatma ve kabuk çalıştırma `MEDIUM` risklidir ve HUD'da ayrı kullanıcı onayı
olmadan yürütülmez. Bazı yıkıcı komut kalıpları onay verilse bile reddedilir.

Kabuk filtresi eksiksiz bir güvenlik duvarı değildir. Asıl sınır kullanıcı onayı,
çalışma alanı kapsamı ve uygulamayı yönetici olmayan hesapta çalıştırmaktır.

## Otonomi

Otonom ajanların gerçek çalışma alanına yazma ve kabuk yetkisi yoktur. Bu yetkiler
yalnız geçici sandbox içinde verilir. Deney sonucu kaynak paketine kendiliğinden
terfi ettirilmez ve ağ erişimi varsayılan olarak kapalıdır.

## Veri ve ağ

Kalıcı hafıza, konuşma veritabanı, belgeler, günlükler ve sırlar yerel runtime
klasörlerinde tutulur ve Git dışında bırakılır. Yerel Ollama/voice çağrıları
loopback üzerindedir. Araştırma isteği verildiğinde sorgu ve kaynak URL'leri seçilen
arama sağlayıcılarına gönderilir; indirilen web metni güvenilmeyen veri sayılır.

## Bilinen sınırlar

- Telegram bot anahtarı Windows DPAPI ile mevcut kullanıcı hesabına bağlı olarak
  şifrelenir. Aynı oturumdaki kötü amaçlı süreçlere karşı mutlak koruma sağlamaz.
- `shell.run` bilinçli olarak genel amaçlıdır ve onaylandıktan sonra kullanıcının
  Windows yetkileriyle çalışır.
- Yerel kötü amaçlı süreçlere karşı güçlü izolasyon yoktur.
- LLM çıktıları hatalı olabilir; kullanıcı onayı doğruluk garantisi değildir.
- Uygulama internete açık servis veya çok kullanıcılı ortam için tasarlanmamıştır.
