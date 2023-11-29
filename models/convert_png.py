from PIL import Image

def convert_to_png(input_path, output_path):
    img = Image.open(input_path)
    img = img.convert("RGBA")
    data = img.getdata()

    new_data = []
    for item in data:
        # 픽셀이 검은색이면 투명하게 만들기
        if item[0] < 30 and item[1] < 30 and item[2] < 30:
            new_data.append((0, 0, 0, 0))
        else:
            new_data.append(item)

    # 새로운 픽셀 데이터를 이미지에 설정
    result_img = Image.new("RGBA", img.size, (0, 0, 0, 0))
    result_img.putdata(new_data)

    # PNG로 저장
    result_img.save(output_path, "PNG")

if __name__ == "__main__":
    input_path = "/root/MultiDreamer/data/assets/sheep_and_lamp/1_segmented_sheep.jpg"  # 입력 JPEG 파일 경로
    output_path = "/root/MultiDreamer/data/assets/sheep_and_lamp/1_segmented_sheep.jpg"  # 출력 PNG 파일 경로

    convert_to_png(input_path, output_path)
