# Güvenlik Politikası

## Açık bildirme

Güvenlik açığını herkese açık issue olarak paylaşmayın. Depo sahibi GitHub'da
**Settings → Security → Private vulnerability reporting** özelliğini açtıysa
**Security → Report a vulnerability** yolunu kullanın. Bu özellik açık değilse
depo sahibinin profilindeki özel iletişim kanalını kullanın.

Bildirimde etkilenen sürüm/commit, yeniden üretme adımları, beklenen etki ve varsa
asgari kanıt yer almalıdır. Gerçek kullanıcı verisi veya aktif erişim anahtarı
eklemeyin.

## Destek kapsamı

Yalnız varsayılan `main` dalının son sürümü güvenlik düzeltmeleri alır. ZESTOLES
tek kullanıcılı, loopback'e bağlı yerel masaüstü yazılımıdır; internete açık sunucu
olarak çalıştırılması desteklenmez.

## Anahtar sızıntısı

Bir Telegram ya da GitHub anahtarı commit'e girdiyse dosyadan silmek yeterli
değildir. Anahtarı sağlayıcısında derhal iptal edin, yenisini üretin ve Git
geçmişini ayrıca temizleyin.
