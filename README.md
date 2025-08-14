# python_pseudocode_converter
tools converter dari kode python ke pseudocode

# Cara Penggunaan
##### buat file contoh.py
```sh
def hitung(a, b):
    if a > b:
        return a - b
    else:
        return b - a
print(hitung(10, 5))
```

##### jalankan dengan command pada terminal dengan perintah 
```sh
python py2pseudo.py contoh.py
```
##### Nanti hasilnya seperti ini :
```sh
PSEUDOCODE START
    FUNCTION hitung(a, b):
        IF (a > b) THEN
            RETURN (a - b)
        ELSE
            RETURN (b - a)
        END IF
    END FUNCTION hitung
    DO print(hitung(10, 5))
PSEUDOCODE END
```
##### Cara meyimpan langsung berbentuk file txt
```sh
python py2pseudo.py contoh.py -o hasil.txt
```
##### Menampilkan nomor baris asli
```sh
python py2pseudo.py contoh.py --show-lineno
```
