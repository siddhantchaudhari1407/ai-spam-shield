import requests

def get_external_ip():
    try:
        response = requests.get('https://api.ipify.org?format=json')
        ip = response.json()['ip']
        print("\n" + "="*40)
        print("🌍 YOUR EXTERNAL IP ADDRESS")
        print("="*40)
        print(f"IP: {ip}")
        print("="*40)
        print("\nNote: When your friend opens the Localtunnel link,")
        print("they might need to enter this IP to verify the tunnel.")
        print("="*40 + "\n")
    except Exception as e:
        print(f"Error fetching IP: {e}")

if __name__ == "__main__":
    get_external_ip()
