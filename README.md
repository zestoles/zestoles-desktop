# ZESTOLES

[![CI](https://github.com/zestoles/zestoles-desktop/actions/workflows/ci.yml/badge.svg)](https://github.com/zestoles/zestoles-desktop/actions/workflows/ci.yml)
[![CodeQL](https://github.com/zestoles/zestoles-desktop/actions/workflows/codeql.yml/badge.svg)](https://github.com/zestoles/zestoles-desktop/actions/workflows/codeql.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

ZESTOLES, Windows üzerinde yerel modellerle çalışan Türkçe sesli masaüstü
asistanıdır. Canlı HUD, doğal konuşma, araç kullanımı, kalıcı hafıza, kaynaklı
araştırma, ajan görevleri, hatırlatıcılar ve isteğe bağlı Telegram denetimi tek
uygulamada birleşir.

> [!WARNING]
> Bu yazılım dosya yazabilir, uygulama çalıştırabilir ve kullanıcı onayından sonra
> kabuk komutu yürütebilir. Yalnız güvendiğiniz kodu çalıştırın; ZESTOLES'i yönetici
> olarak başlatmayın ve `127.0.0.1` bağını internete açmayın.

## Özellikler

- Türkçe STT: `faster-whisper large-v3-turbo`
- Türkçe TTS: Chatterbox Multilingual V3
- Yerel LLM: `qwen3.5:9b`; ağır görev modeli: `qwen3:14b`
- Yerel embedding: `bge-m3`, SQLite ve Markdown hafıza kasası
- Yarı çift yönlü ses: ZESTOLES konuşurken mikrofon girdisi işlenmez
- Kaynak okuma, prompt-injection temizleme ve bağımsız kaynak doğrulama
- Çok adımlı ajan görevleri ve ayrı sonuç doğrulaması
- Dosya, pano, uygulama, süreç ve güvenli komut araçları
- Tek kullanıcılı Telegram eşleştirmesi ve yerel onay akışı
- Animasyonlu HUD, yazı/ses modları, görevler ve ayarlar paneli

## Gereksinimler

- Windows 10 veya 11
- Python 3.12 ortamını kurabilen [`uv`](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com/)
- Git
- Docker Desktop (yalnız isteğe bağlı yerel SearXNG araması için)
- NVIDIA GPU ve güncel sürücü; doğal ses için 12 GB+ VRAM önerilir
- Yaklaşık 20 GB boş disk alanı (modeller ve iki Python ortamı dahil)

## Kurulum

Depoyu klonlayın ve kökteki `KURULUM.cmd` dosyasını çalıştırın:

```powershell
git clone https://github.com/zestoles/zestoles-desktop.git
cd zestoles-desktop
.\KURULUM.cmd
```

Kurulum; Python ortamlarını oluşturur, sabitlenmiş Chatterbox commit'ini ve yerel
modelleri indirir, testleri çalıştırır ve masaüstü kısayolunu kurar. Büyük model
indirmeleri nedeniyle ilk kurulum uzun sürebilir.

Kişiselleştirmek için `config.json` içindeki `user.name` değerini değiştirin.
Bot anahtarlarını veya başka sırları bu dosyaya yazmayın.

### İsteğe bağlı yerel SearXNG

Docker Desktop kuruluysa yerel arama sağlayıcısını şu şekilde başlatabilirsiniz:

```powershell
cd tools\searxng
.\start.ps1
```

İlk çalıştırma `tools/searxng/.env` içinde rastgele bir yerel sır üretir. Bu dosya
Git tarafından yok sayılır. Servis yalnız `127.0.0.1:8888` adresine bağlanır.
Kapatmak için aynı klasörde `docker compose down` çalıştırın.

## Kullanım

Masaüstündeki **ZESTOLES** kısayoluna veya `ZESTOLES.cmd` dosyasına çift
tıklayın. **Ctrl+Alt+J** pencereyi öne getirir. HUD'daki **Kapat** düğmesi ya da
tepsi menüsü tüm ZESTOLES alt süreçlerini güvenli biçimde kapatır.

```powershell
.\ZESTOLES.cmd
.\.venv\Scripts\python.exe run.py --durum
.\.venv\Scripts\python.exe run.py --arastir "araştırma sorusu"
.\.venv\Scripts\python.exe run.py --ajan "tamamlanacak hedef"
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

## Telegram

HUD'daki Telegram bölümünden BotFather anahtarını girin, altı haneli eşleştirme
kodu üretin ve botunuza `/pair KOD` gönderin. Anahtar ve sahip sohbet kimliği
`data/secrets/telegram.json` içinde Windows DPAPI ile kullanıcı hesabına bağlı
olarak şifrelenir; bu klasör Git tarafından yok sayılır. Bir anahtar yanlışlıkla
paylaşılırsa BotFather üzerinden hemen iptal edip yenisini üretin.

## Gizlilik ve güvenlik

- Ana servis varsayılan olarak yalnız `127.0.0.1:8797` adresine bağlanır.
- Değiştirici HUD istekleri her çalıştırmada üretilen geçici belirteçle korunur.
- Yazma, taşıma, uygulama açma/kapatma ve `shell.run` kullanıcı onayı ister.
- Otonom ajanlar gerçek dosya sistemi yazma ve kabuk yetkisini yalnız sandbox
  içinde alabilir; kaynak ağacına otomatik terfi kapalıdır.
- Hafıza, günlükler, belgeler, veritabanları, ses modelleri ve Telegram sırları
  `.gitignore` kapsamındadır.
- Web araştırması dışındaki sohbet/ses/hafıza akışı yereldir. Araştırma yapılınca
  sorgular ve kaynak istekleri ilgili internet servislerine gider.

Tehdit modeli ve bilinen sınırlar için [güvenlik modeline](docs/SECURITY-MODEL.md),
bir açık bildirmek için [SECURITY.md](SECURITY.md) dosyasına bakın.

## Lisans

ZESTOLES, [MIT Lisansı](LICENSE) altında açık kaynak olarak yayımlanır. Yazılımı
ticari veya özel amaçla kullanabilir, değiştirebilir ve dağıtabilirsiniz; telif
hakkı ve lisans bildirimlerini korumanız gerekir. Yazılım garanti verilmeden
sunulur.
