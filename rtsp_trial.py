import rtsp


def process_image(image):
    return image.resize([640, 640])


if __name__ == '__main__':

    rstp_url = 'rtsp://admin:Mp010201@10.111.45.211:554/Streaming/channels/101'
    with rtsp.Client(rstp_url) as client:
        print(client.type)
        image = client.read()
        while True:
            process_image(image).show()
            client.read(raw=True)

