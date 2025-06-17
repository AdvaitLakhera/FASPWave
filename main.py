import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import sounddevice as sd
import numpy as np
import codec
import time
import os
import soundfile as sf

class TinkerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Audio Modem")
        self.root.geometry("1200x800")
        self.root.configure(bg='#0f0f23')

        # Current protocol settings
        self.current_mode = tk.StringVar(value="FAST")
        self.custom_carriers = tk.StringVar(value="4000-18000")
        self.custom_symbol_rate = tk.StringVar(value="100")
        self.custom_bits = tk.StringVar(value="4")
        self.custom_rs = tk.StringVar(value="20")

        # Audio visualization
        self.audio_bars = []
        self.is_transmitting = False
        self.is_receiving = False

        self.setup_ui()
        self.setup_protocol()

    def setup_ui(self):
        # Main container
        main_frame = tk.Frame(self.root, bg='#0f0f23')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Header
        header = tk.Frame(main_frame, bg='#1a1a2e', relief=tk.RAISED, bd=2)
        header.pack(fill=tk.X, pady=(0, 20))

        title = tk.Label(header, text="🔊 TinkerGUI Audio Modem",
                         font=('Arial', 24, 'bold'), fg='#00d4ff', bg='#1a1a2e')
        title.pack(pady=15)

        # Main content in two columns
        content_frame = tk.Frame(main_frame, bg='#0f0f23')
        content_frame.pack(fill=tk.BOTH, expand=True)

        # Left panel - Protocol & Controls
        left_panel = tk.LabelFrame(content_frame, text="Protocol & Controls",
                                   font=('Arial', 14, 'bold'), fg='#00d4ff', bg='#1a1a2e')
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        # Protocol selection
        protocol_frame = tk.Frame(left_panel, bg='#1a1a2e')
        protocol_frame.pack(fill=tk.X, padx=10, pady=10)

        tk.Label(protocol_frame, text="Protocol Mode:", font=('Arial', 12, 'bold'),
                 fg='white', bg='#1a1a2e').pack(anchor=tk.W)

        modes = ["FAST", "FASTEST", "SUBMARINE", "LIGHTNING", "ULTRASOUND", "CUSTOM"]
        mode_combo = ttk.Combobox(protocol_frame, textvariable=self.current_mode,
                                  values=modes, state="readonly", width=20)
        mode_combo.pack(anchor=tk.W, pady=5)
        mode_combo.bind('<<ComboboxSelected>>', self.on_mode_change)

        # Custom protocol settings
        self.custom_frame = tk.LabelFrame(left_panel, text="Custom Protocol",
                                          font=('Arial', 12, 'bold'), fg='#7c3aed', bg='#1a1a2e')
        self.custom_frame.pack(fill=tk.X, padx=10, pady=10)

        # Custom settings grid
        settings = [
            ("Carrier Range (Hz):", self.custom_carriers, "4000-18000"),
            ("Symbol Rate:", self.custom_symbol_rate, "100"),
            ("Bits/Symbol:", self.custom_bits, "4"),
            ("Reed-Solomon:", self.custom_rs, "20")
        ]

        for i, (label, var, default) in enumerate(settings):
            tk.Label(self.custom_frame, text=label, fg='white', bg='#1a1a2e').grid(
                row=i, column=0, sticky=tk.W, padx=5, pady=2)
            entry = tk.Entry(self.custom_frame, textvariable=var, width=15, bg='#2d2d44', fg='white')
            entry.grid(row=i, column=1, padx=5, pady=2)

        # Apply custom button
        apply_btn = tk.Button(self.custom_frame, text="Apply Custom", command=self.apply_custom,
                              bg='#7c3aed', fg='white', font=('Arial', 10, 'bold'))
        apply_btn.grid(row=len(settings), column=0, columnspan=2, pady=10)

        # Message input
        msg_frame = tk.Frame(left_panel, bg='#1a1a2e')
        msg_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        tk.Label(msg_frame, text="Message:", font=('Arial', 12, 'bold'),
                 fg='white', bg='#1a1a2e').pack(anchor=tk.W)

        self.message_text = scrolledtext.ScrolledText(msg_frame, height=8, width=40,
                                                      bg='#2d2d44', fg='white', font=('Consolas', 10))
        self.message_text.pack(fill=tk.BOTH, expand=True, pady=5)

        # Control buttons
        btn_frame = tk.Frame(left_panel, bg='#1a1a2e')
        btn_frame.pack(fill=tk.X, padx=10, pady=10)

        buttons = [
            ("📤 Transmit", self.transmit_message, '#00d4ff'),
            ("📥 Receive", self.receive_message, '#f59e0b'),
            ("📁 Load File", self.load_file, '#10b981'),
            ("💾 Save File", self.save_file, '#8b5cf6')
        ]

        for i, (text, cmd, color) in enumerate(buttons):
            btn = tk.Button(btn_frame, text=text, command=cmd, bg=color, fg='white',
                            font=('Arial', 10, 'bold'), width=12)
            btn.grid(row=i // 2, column=i % 2, padx=5, pady=5, sticky=tk.EW)

        btn_frame.grid_columnconfigure(0, weight=1)
        btn_frame.grid_columnconfigure(1, weight=1)

        # Right panel - Visualization & Status
        right_panel = tk.LabelFrame(content_frame, text="Audio Visualization & Status",
                                    font=('Arial', 14, 'bold'), fg='#00d4ff', bg='#1a1a2e')
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))

        # Audio visualization
        viz_frame = tk.Frame(right_panel, bg='#1a1a2e')
        viz_frame.pack(fill=tk.X, padx=10, pady=10)

        tk.Label(viz_frame, text="Audio Spectrum:", font=('Arial', 12, 'bold'),
                 fg='white', bg='#1a1a2e').pack(anchor=tk.W)

        self.canvas = tk.Canvas(viz_frame, width=400, height=200, bg='#0f0f23', highlightthickness=0)
        self.canvas.pack(pady=10)

        # Progress bars
        progress_frame = tk.Frame(right_panel, bg='#1a1a2e')
        progress_frame.pack(fill=tk.X, padx=10, pady=10)

        tk.Label(progress_frame, text="Transmission Progress:", font=('Arial', 12, 'bold'),
                 fg='white', bg='#1a1a2e').pack(anchor=tk.W)

        self.tx_progress = ttk.Progressbar(progress_frame, mode='determinate', length=380)
        self.tx_progress.pack(pady=5)

        tk.Label(progress_frame, text="Reception Progress:", font=('Arial', 12, 'bold'),
                 fg='white', bg='#1a1a2e').pack(anchor=tk.W)

        self.rx_progress = ttk.Progressbar(progress_frame, mode='determinate', length=380)
        self.rx_progress.pack(pady=5)

        # Status log
        status_frame = tk.Frame(right_panel, bg='#1a1a2e')
        status_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        tk.Label(status_frame, text="Status Log:", font=('Arial', 12, 'bold'),
                 fg='white', bg='#1a1a2e').pack(anchor=tk.W)

        self.status_log = scrolledtext.ScrolledText(status_frame, height=12, width=50,
                                                    bg='#2d2d44', fg='#00ff00', font=('Consolas', 9))
        self.status_log.pack(fill=tk.BOTH, expand=True, pady=5)

        # Initialize visualization
        self.init_visualization()
        self.log_status("TinkerGUI Audio Modem initialized")

    def init_visualization(self):
        """Initialize audio spectrum visualization"""
        self.canvas.delete("all")
        # Draw frequency axis
        for i in range(20):
            x = 20 + i * 18
            height = np.random.randint(10, 150)
            color = f"#{int(255 - height):02x}{int(height):02x}ff"
            self.canvas.create_rectangle(x, 180, x + 15, 180 - height,
                                         fill=color, outline=color)

    def setup_protocol(self):
        """Setup initial protocol"""
        codec.setup_mode_params("FAST")
        self.log_status("Protocol set to FAST mode")

    def on_mode_change(self, event=None):
        """Handle protocol mode change"""
        mode = self.current_mode.get()
        if mode != "CUSTOM":
            codec.setup_mode_params(mode)
            self.log_status(f"Protocol changed to {mode} mode")

    def apply_custom(self):
        """Apply custom protocol settings"""
        try:
            # Parse carrier range
            carrier_range = self.custom_carriers.get()
            start_freq, end_freq = map(int, carrier_range.split('-'))
            num_carriers = 16  # Default

            # Create custom carrier frequencies
            custom_carriers = np.linspace(start_freq, end_freq, num_carriers, dtype=np.float32)

            # Apply custom settings to codec
            codec.CARRIER_FREQS = custom_carriers
            codec.SYMBOL_RATE = int(self.custom_symbol_rate.get())
            codec.BITS_PER_SYMBOL = int(self.custom_bits.get())

            # Create Reed-Solomon codec
            from reedsolo import RSCodec
            codec.RS_CODEC = RSCodec(int(self.custom_rs.get()))

            self.log_status(f"Custom protocol applied: {carrier_range} Hz, "
                            f"{codec.SYMBOL_RATE} sym/s, {codec.BITS_PER_SYMBOL} bits/sym")

        except Exception as e:
            messagebox.showerror("Error", f"Invalid custom settings: {e}")

    def log_status(self, message):
        """Add message to status log"""
        timestamp = time.strftime("%H:%M:%S")
        self.status_log.insert(tk.END, f"[{timestamp}] {message}\n")
        self.status_log.see(tk.END)

    def update_visualization(self, audio_data=None):
        """Update audio spectrum visualization"""
        self.canvas.delete("all")

        if audio_data is not None and len(audio_data) > 0:
            # Simple spectrum visualization
            fft_data = np.abs(np.fft.rfft(audio_data[:1024]))
            fft_data = fft_data[:20]  # Take first 20 bins
            max_val = np.max(fft_data) if np.max(fft_data) > 0 else 1

            for i, val in enumerate(fft_data):
                x = 20 + i * 18
                height = int((val / max_val) * 150)
                color = f"#00{int(255 * val / max_val):02x}ff"
                self.canvas.create_rectangle(x, 180, x + 15, 180 - height,
                                             fill=color, outline=color)
        else:
            # Default visualization
            self.init_visualization()

    def transmit_message(self):
        """Transmit message using audio modem"""
        if self.is_transmitting:
            return

        message = self.message_text.get(1.0, tk.END).strip()
        if not message:
            messagebox.showwarning("Warning", "Please enter a message to transmit")
            return

        self.is_transmitting = True
        self.tx_progress['value'] = 0

        def tx_thread():
            try:
                self.log_status(f"Encoding message ({len(message)} chars)...")
                self.tx_progress['value'] = 25

                # Encode based on current mode
                mode = self.current_mode.get()
                if mode == "SUBMARINE":
                    audio = codec.encode_submarine_message(message)
                elif mode == "LIGHTNING":
                    audio = codec.encode_lightning_message(message)
                else:
                    audio = codec.encode_message_optimized(message)

                self.tx_progress['value'] = 50
                self.log_status("Playing audio transmission...")
                self.update_visualization(audio)

                # Play audio
                sd.play(audio, codec.SAMPLE_RATE)
                self.tx_progress['value'] = 75
                sd.wait()

                self.tx_progress['value'] = 100
                self.log_status("Transmission complete!")

            except Exception as e:
                self.log_status(f"Transmission error: {e}")
                messagebox.showerror("Error", f"Transmission failed: {e}")
            finally:
                self.is_transmitting = False
                self.root.after(2000, lambda: setattr(self.tx_progress, 'value', 0))

        threading.Thread(target=tx_thread, daemon=True).start()

    def receive_message(self):
        """Receive and decode audio message"""
        if self.is_receiving:
            return

        self.is_receiving = True
        self.rx_progress['value'] = 0

        def rx_thread():
            try:
                duration = 10  # Record for 10 seconds
                self.log_status(f"Recording audio for {duration} seconds...")
                self.rx_progress['value'] = 10

                # Record audio
                audio = sd.rec(int(duration * codec.SAMPLE_RATE),
                               samplerate=codec.SAMPLE_RATE,
                               channels=1,
                               dtype=np.float32)
                sd.wait()  # Wait until recording is finished

                # Convert to numpy array and ensure proper shape
                audio = np.squeeze(audio)  # Remove channel dimension if mono

                # Save to file (WAV format)
                output_filename = "received_audio.wav"
                sf.write(output_filename, audio, codec.SAMPLE_RATE)
                self.log_status(f"Audio saved to {output_filename}")
                self.rx_progress['value'] = 100

                self.rx_progress['value'] = 50
                self.log_status("Decoding received audio...")
                self.update_visualization(audio.flatten())

                # Decode based on current mode
                mode = self.current_mode.get()
                if mode == "SUBMARINE":
                    decoded = codec.decode_submarine_audio(audio.flatten())
                else:
                    decoded = codec.decode_audio_optimized(audio.flatten())

                self.rx_progress['value'] = 100

                if decoded and not decoded.startswith("❌"):
                    self.message_text.delete(1.0, tk.END)
                    self.message_text.insert(1.0, decoded)
                    self.log_status("Message decoded successfully!")
                else:
                    self.log_status(f"Decode failed: {decoded}")

            except Exception as e:
                self.log_status(f"Reception error: {e}")
                messagebox.showerror("Error", f"Reception failed: {e}")
            finally:
                self.is_receiving = False
                self.root.after(2000, lambda: setattr(self.rx_progress, 'value', 0))

        threading.Thread(target=rx_thread, daemon=True).start()

    def load_file(self):
        """Load text file"""
        filename = filedialog.askopenfilename(
            title="Load Text File",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if filename:
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    content = f.read()
                self.message_text.delete(1.0, tk.END)
                self.message_text.insert(1.0, content)
                self.log_status(f"Loaded file: {os.path.basename(filename)}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file: {e}")

    def save_file(self):
        """Save message to text file"""
        filename = filedialog.asksaveasfilename(
            title="Save Text File",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if filename:
            try:
                content = self.message_text.get(1.0, tk.END).strip()
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(content)
                self.log_status(f"Saved file: {os.path.basename(filename)}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save file: {e}")


def main():
    root = tk.Tk()
    app = TinkerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()