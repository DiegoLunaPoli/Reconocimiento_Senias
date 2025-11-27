import os
from moviepy.editor import VideoFileClip

# -------------------------------------------------------------------
# CONFIGURACIÓN
# -------------------------------------------------------------------

INPUT_DIR  = r"C:\Users\julia\OneDrive PolitecnicoGrancolombiano\Documentos\U\SEMESTRE 6\SISTEMAS OPERACIONALES\PROG\proyecto\Reconocimiento_Senias\videos"
OUTPUT_DIR = r"C:\Users\julia\OneDrive PolitecnicoGrancolombiano\Documentos\U\SEMESTRE 6\SISTEMAS OPERACIONALES\PROG\proyecto\Reconocimiento_Senias\videos_proc"

# -------------------------------------------------------------------
# DATOS: tiempos por letra y video
# -------------------------------------------------------------------

videos_a_procesar = [

    # -------------------------------------------------------------
    # Abecedario - Lengua de Señas Colombiana - LSC
    # -------------------------------------------------------------
    *[
        {
            "filename": "Abecedario - Lengua de Señas Colombiana - LSC .webm",
            "letter": letter,
            "cuts": [(start, end)]
        }
        for letter, (start, end) in zip(
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
            [
                (0, 3), (4, 7), (8, 9), (11, 13), (14, 17), (18, 21),
                (22, 24), (26, 28), (29, 30), (31, 32), (35, 37), (38, 40),
                (45, 47), (48, 51), (57, 60), (61, 63), (65, 68), (69, 71),
                (76, 78), (80, 82), (83, 85), (85, 87), (88, 90), (91, 93),
                (95, 98), (99, 101)
            ]
        )
    ],

    # -------------------------------------------------------------
    # Abecedario (Abc) – este no tiene tiempos en tu tabla
    # -------------------------------------------------------------

    # Si quieres agregarlo luego, puedo generarlo también.


    # -------------------------------------------------------------
    # Abecedario en Lengua de Señas colombianas - LSC.f136
    # -------------------------------------------------------------
    *[
        {
            "filename": "Abecedario en Lengua de Señas colombianas - LSC.f136.mp4",
            "letter": letter,
            "cuts": [(start, end)]
        }
        for letter, (start, end) in zip(
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
            [
                (12, 15), (16, 20), (23, 24), (26, 28), (30, 32), (35, 38),
                (45, 47), (50, 54), (57, 59), (63, 64), (67, 74), (76, 79),
                (83, 84), (85, 92), (101, 103), (105, 109), (111, 117),
                (120, 122), (123, 124), (129, 131), (133, 136), (138, 139),
                (140, 141), (142, 143), (144, 146), (147, 149)
            ]
        )
    ],

    # -------------------------------------------------------------
    # Alfabeto de Lengua de Señas Mexicana
    # -------------------------------------------------------------
    *[
        {
            "filename": "Alfabeto de Lengua de Señas Mexicana.webm",
            "letter": letter,
            "cuts": [(start, end)]
        }
        for letter, (start, end) in zip(
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
            [
                (5, 7), (8, 10), (11, 12), (13, 16), (17, 19), (20, 22),
                (23, 26), (26, 28), (29, 31), (32, 34), (35, 37), (38, 40),
                (43, 46), (47, 49), (53, 56), (56, 59), (59, 61), (62, 65),
                (68, 70), (71, 73), (74, 76), (77, 79), (80, 82), (83, 85),
                (86, 88), (89, 91)
            ]
        )
    ],

    # -------------------------------------------------------------
    # El abecedario en lengua de señas colombiana.f135
    # -------------------------------------------------------------
    *[
        {
            "filename": "El abecedario en lengua de señas colombiana.f135.mp4",
            "letter": letter,
            "cuts": [(start, end)]
        }
        for letter, (start, end) in zip(
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
            [
                (13, 19), (20, 24), (25, 31), (32, 37), (38, 43), (44, 48),
                (49, 54), (55, 60), (62, 66), (67, 70), (72, 77), (78, 82),
                (87, 92), (93, 98), (105, 109), (110, 114), (115, 119),
                (120, 125), (130, 136), (137, 140), (142, 146), (148, 151),
                (153, 156), (157, 161), (162, 166), (167, 171)
            ]
        )
    ],

    # -------------------------------------------------------------
    # El Abecedario en lengua de señas colombiana.f136
    # -------------------------------------------------------------
    *[
        {
            "filename": "El Abecedario en lengua de señas colombiana.f136.mp4",
            "letter": letter,
            "cuts": [(start, end)]
        }
        for letter, (start, end) in zip(
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
            [
                (0, 3), (4, 5), (6, 8), (9, 11), (12, 15), (16, 18),
                (19, 21), (23, 26), (26, 29), (30, 36), (35, 37), (38, 40),
                (41, 43), (46, 48), (52, 55), (56, 59), (60, 63), (64, 66),
                (67, 70), (71, 73), (74, 76), (77, 80), (82, 83), (85, 88),
                (89, 91), (92, 96)
            ]
        )
    ],
]

# -------------------------------------------------------------------
# UTILIDADES DE RECORTE
# -------------------------------------------------------------------

def convertir_a_segundos(t):
    if isinstance(t, (int, float)):
        return t
    if isinstance(t, str) and ":" in t:
        minutos, segundos = t.split(":")
        return int(minutos) * 60 + int(segundos)
    return int(t)

def siguiente_indice_clip(folder):
    existentes = [
        f for f in os.listdir(folder)
        if f.startswith("clip_") and f.endswith(".mp4")
    ]
    if not existentes:
        return 1
    numeros = []
    for f in existentes:
        try:
            numeros.append(int(f.replace("clip_", "").replace(".mp4", "")))
        except:
            pass
    return max(numeros) + 1

def recortar_video(input_path, output_folder, cuts):
    print(f"Procesando video: {input_path}")
    video = VideoFileClip(input_path)
    indice_clip = siguiente_indice_clip(output_folder)

    for start, end in cuts:
        output_path = os.path.join(output_folder, f"clip_{indice_clip}.mp4")
        indice_clip += 1
        print(f"Generando clip {output_path}: {start}s a {end}s")
        clip = video.subclip(start, end)
        clip.write_videofile(output_path, codec="libx264")

    video.close()

# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for item in videos_a_procesar:
        filename = item["filename"]
        letter   = item["letter"]
        cuts     = item["cuts"]

        print("\n============================")
        print(f"Procesando letra: {letter}")
        print(f"Video: {filename}")

        input_path = os.path.join(INPUT_DIR, filename)
        letter_folder = os.path.join(OUTPUT_DIR, letter)
        os.makedirs(letter_folder, exist_ok=True)

        recortar_video(input_path, letter_folder, cuts)

    print("\nProceso completado")

if __name__ == "__main__":
    main()