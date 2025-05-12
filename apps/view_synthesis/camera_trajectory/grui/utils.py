import random as rd
import socket


def find_available_ports(count: int = 1, 
                    start_port: int = 1024, 
                      end_port: int = 65535):
    """
    Find a list of available ports on localhost.

    Args:
        count (int): Number of ports to find.
        start_port (int): Starting port in the range to check.
        end_port (int): Ending port in the range to check.

    Returns:
        List[int]: A list of available port numbers.
    """
    available_ports = []
    tried_ports = set()
    for port in range(start_port, end_port):
        if len(available_ports) >= count:
            break
        if port in tried_ports:
            continue
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('localhost', port))
                s.listen(1)
                available_ports.append(port)
            except OSError:
                continue
            finally:
                tried_ports.add(port)
    return available_ports


# Make sure that gradio uses dark theme.
_APP_JS = """
function refresh() {
    const url = new URL(window.location);
    if (url.searchParams.get('__theme') !== 'dark') {
        url.searchParams.set('__theme', 'dark');
    }
}
"""

USER_GUIDE = """
---
## 🛠️ User Guide:

### 1️⃣ Start the Server
- Click the **`Run Viser`** button to initialize and launch the Viser server.

### 2️⃣ Select Multi-view
- In the **`Examples`** section:
  - Choose your desired image. It will expand the relative multi-view.
  - Click **`Confirm`** to proceed.
- Then, click **`Process multi-view`** to begin processing the selected image.

### 3️⃣ Viser Interaction
- For detailed instructions and tips on using Viser, click [**Viser Interaction**](https://github.com/Stability-AI/stable-virtual-camera/blob/main/docs/GR_USAGE.md#advanced)

#### 4️⃣ Save Your Setup
- Once the camera trajectory is already set, click the **`Save data`** button to store your configuration and processed results.
"""

attention_catcher = """
<div style="border: 2px solid #f39c12; background-color: #fffbe6; padding: 16px; border-radius: 8px; font-weight: bold; color: #c0392b; font-size: 13px; text-align: center;">
    👇 Click the Button start server
</div>
"""

