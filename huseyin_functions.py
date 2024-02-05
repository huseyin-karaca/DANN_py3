def get_run_name():
    with open("/home/huseyin/fungtion/DANN_py3/ilveilceler.txt", "r+", encoding="utf-8") as file:
        lines = file.readlines()
        if lines:
            character_name = lines[0].strip()
            file.seek(0)  # Dosyanın başına dön
            for line in lines[1:]:
                file.write(line)  # İlk satırı sil
            file.truncate()  # Dosyanın geri kalanını kes
            file.write(character_name + "\n")  # Karakteri dosyanın sonuna ekle
            return character_name
        else:
            return None
