# Otomatik başlatma

JARVIS'in oturum açılışında kendiliğinden başlaması için bir Görev Zamanlayıcı
girdisi. S9 soak testinde ölçülen sorunu kapatır: Windows 06:30'da güncelleme
için makineyi yeniden başlattı, JARVIS'i geri getiren bir şey olmadığı için
sistem üç saat kapalı kaldı.

## Kurulum

```powershell
powershell -ExecutionPolicy Bypass -File C:\JARVIS\tools\autostart\register.ps1
```

Yönetici gerekmiyor ve bilerek gerekmiyor: görev kullanıcının kendi oturumunda,
kendi yetkisiyle çalışır.

## Durdurma

```powershell
powershell -ExecutionPolicy Bypass -File C:\JARVIS\tools\autostart\stop-jarvis.ps1
```

**`Stop-ScheduledTask` tek başına yetmiyor** — ölçüldü. Görevin eylemi bir
PowerShell başlatıcısı, o cmd'yi, o `python` uygulama takma adını, o da gerçek
yorumlayıcıyı çalıştırıyor. Görev Zamanlayıcı kendi başlattığını sonlandırıyor,
zincirin ucundaki yorumlayıcı çalışmaya devam ediyor: portu ve kilidi tutuyor
ama görev "durdu" görünüyor. `stop-jarvis.ps1` kilit dosyasındaki PID'i esas
alır.

Kapatma zarif değil, kasıtlı olarak: soket kimliksiz olduğu için tel üzerinden
kapatma komutu, herhangi bir yerel sürece verilmiş bir kapatma düğmesi olurdu.
Yarıda kalan görev `running` kalır ve sonraki başlangıçta denemesi sayılarak
kuyruğa döner — kurtarma yolu tam bunun için var.

## Kaldırma

```powershell
powershell -ExecutionPolicy Bypass -File C:\JARVIS\tools\autostart\unregister.ps1
```

## Neden oturum açılışında, servis olarak değil

Politika katmanı otonom işin uygun olup olmadığına `GetLastInputInfo` ile karar
veriyor — klavyede biri var mı. Session 0'da sorulacak bir klavye yok, ölçüm
"bilinmiyor" döner ve politika doğru davranıp hiçbir şey yapmaz. Servis olarak
kurulmuş bir JARVIS, hiç çalışmayan bir JARVIS olurdu.

Bunun bedeli açık: **oturum açılmadan JARVIS çalışmaz.** Makine açık ama kimse
giriş yapmamışsa sistem beklemeye devam eder.

## Ayarlar ve gerekçeleri

| Ayar | Değer | Neden |
|---|---|---|
| Tetikleyici | oturum açılışı + 1 dk gecikme | yeni açılmış bir makine zaten meşgul |
| Çalışma süresi sınırı | yok | aylarca çalışması bekleniyor |
| Hata sonrası yeniden başlatma | 3 kez, 5 dk arayla | çöken bir döngü kendi kendine dönsün |
| Aynı anda birden fazla örnek | yeni olan çalıştırılmaz | ikinci kopya kuyruğu paylaşır, bütçeyi iki kez harcar |
| Pil / boşta durumu | durdurmaz | ne zaman çalışılacağına politika karar verir, Görev Zamanlayıcı değil |

Görev Zamanlayıcı'nın "aynı anda tek örnek" ayarı tek başına yeterli değil:
elle başlatılan bir terminal onun görmediği ikinci bir kopya olurdu. Asıl kilit
`jarvis/cli/instance.py` içindeki PID dosyası — `data/daemon.lock`. Yeniden
başlatmadan sonra bu dosya her zaman bayattır; bayat kilit devralınır, yoksa
otomatik başlatma ilk yeniden başlatmada kendi kendini engellerdi.

## Çalıştığını doğrulamak

```powershell
Get-ScheduledTask -TaskName 'JARVIS Otonom' | Get-ScheduledTaskInfo
```

```powershell
Start-ScheduledTask -TaskName 'JARVIS Otonom'    # yeniden başlatmayı beklemeden dene
```

Sonra: `python run.py --gorevler` ve `python run.py --olaylar`. Günlük
`logs\daemon.out.log` içinde, 5 MB'ta bir devrediyor.

## Elle başlatılan kopya ne olur

`run.py --otonom` ikinci kez başlatılırsa kilidi göremez değil, görür ve
çıkar:

```
otonom döngü zaten çalışıyor (PID 21016) — ikinci bir kopya başlatılmadı
```

Sohbet arayüzü (`python run.py`) de aynı kilide bakar. Daemon çalışıyorken sohbet
açarsan konuşma, hafıza ve araştırma normal çalışır ama sohbet **kendi
zamanlayıcısını başlatmaz**:

```
otonom döngü başka bir süreçte çalışıyor (PID 21016) — görevler oraya kuyruklanır
```

Kuyruk paylaşıldığı için `/gorev` ile eklediğin iş kaybolmaz; çalışan daemon bir
tick içinde alır.
