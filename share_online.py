"""
PharmacoRisk AI - One-Click Share & Run Script
Starts the web application and displays the exact links to open on other devices.
"""

import socket
import os
import sys

def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return '127.0.0.1'

if __name__ == '__main__':
    local_ip = get_local_ip()
    port = 5000
    
    print("=" * 65)
    print("      PHARMCORISK AI - WEB SERVER RUNNING")
    print("=" * 65)
    print()
    print(f"[*] On YOUR computer, open:")
    print(f"    👉 http://127.0.0.1:{port}")
    print(f"    👉 http://localhost:{port}")
    print()
    print(f"[*] To open on OTHER laptops, desktops, or phones on the same Wi-Fi:")
    print(f"    👉 http://{local_ip}:{port}")
    print()
    print("=" * 65)
    print("Press Ctrl+C to stop the server anytime.")
    print("=" * 65)
    print()
    
    # Import and run Flask app
    from app import app
    app.run(host='0.0.0.0', port=port, debug=False)
