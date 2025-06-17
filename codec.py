import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.signal import lfilter, butter, filtfilt
import zlib
from reedsolo import RSCodec
import time
from numba import jit, njit, prange

SAMPLE_RATE = 48000
VOLUME = 1
PREAMBLE_SYMBOLS = np.array([0, 1, 2, 3], dtype=np.int32)
POSTAMBLE_SYMBOLS = np.array([3, 2, 1, 0], dtype=np.int32)


# ================== MODE CONFIGURATION ==================
def setup_mode_params(mode):
    global CARRIER_FREQS, SYMBOL_RATE, BITS_PER_SYMBOL, RS_CODEC, \
        PREAMBLE_SYMBOLS, POSTAMBLE_SYMBOLS

    if mode == "FAST":
        CARRIER_FREQS = np.linspace(4000, 18000, 16, dtype=np.float32)
        SYMBOL_RATE = 100
        BITS_PER_SYMBOL = 4
        RS_CODEC = RSCodec(20)
        PREAMBLE_SYMBOLS = np.array([0, 1, 2, 3], dtype=np.int32)
        POSTAMBLE_SYMBOLS = np.array([3, 2, 1, 0], dtype=np.int32)

    elif mode == "FASTEST":
        CARRIER_FREQS = np.linspace(5000, 21000, 20, dtype=np.float32)
        SYMBOL_RATE = 200
        BITS_PER_SYMBOL = 4
        RS_CODEC = RSCodec(25)
        PREAMBLE_SYMBOLS = np.array([0, 1, 2, 3], dtype=np.int32)
        POSTAMBLE_SYMBOLS = np.array([3, 2, 1, 0], dtype=np.int32)

    elif mode == "SUBMARINE":
        CARRIER_FREQS = np.linspace(2000, 24000, 32, dtype=np.float32)
        SYMBOL_RATE = 150
        BITS_PER_SYMBOL = 4
        RS_CODEC = RSCodec(80)
        # Enhanced synchronization patterns
        PREAMBLE_SYMBOLS = np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1], dtype=np.int32)
        POSTAMBLE_SYMBOLS = np.array([3, 2, 1, 0, 3, 2, 1, 0, 3, 2], dtype=np.int32)

    elif mode == "LIGHTNING":
        CARRIER_FREQS = np.linspace(2000, 28000, 32, dtype=np.float32)
        SYMBOL_RATE = 800
        BITS_PER_SYMBOL = 4
        RS_CODEC = RSCodec(40)
        PREAMBLE_SYMBOLS = np.array([0, 1, 2, 3], dtype=np.int32)
        POSTAMBLE_SYMBOLS = np.array([3, 2, 1, 0], dtype=np.int32)

    elif mode == "ULTRASOUND":
        CARRIER_FREQS = np.linspace(16000, 19000, 32, dtype=np.float32)
        SYMBOL_RATE = 100
        BITS_PER_SYMBOL = 4
        RS_CODEC = RSCodec(40)
        PREAMBLE_SYMBOLS = np.array([0, 1, 2, 3], dtype=np.int32)
        POSTAMBLE_SYMBOLS = np.array([3, 2, 1, 0], dtype=np.int32)

    else:
        raise ValueError(f"Invalid mode selected: {mode}")


# ================== SPECIALIZED SUBMARINE MODE ==================
@njit
def generate_submarine_tone(freq, duration, sample_rate, volume, fade_ratio=0.1):
    """Generate a single tone with enhanced envelope for wall penetration"""
    samples = int(duration * sample_rate)
    t = np.linspace(0, duration, samples).astype(np.float32)

    envelope = np.ones(samples, dtype=np.float32)
    fade_samples = int(samples * fade_ratio)

    for i in range(fade_samples):
        envelope[i] = 0.5 - 0.5 * np.cos(np.pi * i / fade_samples)

    for i in range(fade_samples):
        envelope[samples - 1 - i] = 0.5 - 0.5 * np.cos(np.pi * i / fade_samples)

    signal = np.zeros(samples, dtype=np.float32)
    for harmonic in range(1, 4):
        harmonic_amp = volume / (harmonic * harmonic)
        phase = 2 * np.pi * freq * harmonic * t
        signal += harmonic_amp * np.sin(phase)

    return signal * envelope


@njit
def generate_submarine_sequence(symbols, symbol_rate, sample_rate, volume):
    """Generate audio sequence optimized for SUBMARINE mode"""
    duration_per_symbol = 1.0 / symbol_rate
    samples_per_symbol = int(sample_rate * duration_per_symbol)

    total_samples = len(symbols) * samples_per_symbol
    audio_signal = np.zeros(total_samples, dtype=np.float32)

    for i in range(len(symbols)):
        symbol = symbols[i]
        freq_idx = symbol % len(CARRIER_FREQS)
        freq = CARRIER_FREQS[freq_idx]

        # Generate tone for this symbol
        start_idx = i * samples_per_symbol
        tone = generate_submarine_tone(freq, duration_per_symbol, sample_rate, volume)

        # Add to signal
        for j in range(len(tone)):
            if start_idx + j < total_samples:
                audio_signal[start_idx + j] = tone[j]

    return audio_signal


def encode_submarine_message(message: str, repeat_count=3) -> np.ndarray:
    """Encode message for SUBMARINE mode with triple redundancy"""
    print(f"🚢 SUBMARINE MODE: Encoding with {repeat_count}x redundancy")

    # Compress and prepare data
    data = zlib.compress(message.encode(), level=9)
    pad_length = (16 - (len(data) % 16)) % 16
    data_padded = data + bytes([0] * pad_length)

    # Apply strong error correction
    data_encoded = RS_CODEC.encode(data_padded)
    print(f"📦 Data: {len(data)} bytes → {len(data_encoded)} bytes (RS protected)")

    # Convert to symbols
    symbols = bytes_to_symbols_vectorized(np.array(data_encoded, dtype=np.uint8), BITS_PER_SYMBOL)

    # Create transmission with repetition
    transmission_parts = []

    # Long silence at start
    silence_long = np.zeros(int(SAMPLE_RATE * 1.0), dtype=np.float32)
    transmission_parts.append(silence_long)

    for repeat in range(repeat_count):
        print(f"🔄 Generating transmission {repeat + 1}/{repeat_count}")

        # Generate preamble
        preamble_audio = generate_submarine_sequence(PREAMBLE_SYMBOLS, SYMBOL_RATE, SAMPLE_RATE, VOLUME)

        # Generate message
        message_audio = generate_submarine_sequence(symbols, SYMBOL_RATE, SAMPLE_RATE, VOLUME)

        # Generate postamble
        postamble_audio = generate_submarine_sequence(POSTAMBLE_SYMBOLS, SYMBOL_RATE, SAMPLE_RATE, VOLUME)

        # Add inter-transmission silence
        silence_short = np.zeros(int(SAMPLE_RATE * 0.5), dtype=np.float32)

        transmission_parts.extend([preamble_audio, message_audio, postamble_audio, silence_short])

    # Final long silence
    transmission_parts.append(silence_long)

    # Concatenate all parts
    audio_signal = np.concatenate(transmission_parts)

    # Apply final processing for wall penetration
    audio_signal = enhance_for_penetration(audio_signal)

    duration = len(audio_signal) / SAMPLE_RATE
    bitrate = len(data_encoded) * 8 / duration
    print(f"⏱️  Duration: {duration:.2f}s, Effective bitrate: {bitrate:.1f} bps")

    return audio_signal


def enhance_for_penetration(audio_signal):
    """Apply processing to enhance wall penetration"""
    # Apply gentle compression to boost weaker signals
    compressed = np.tanh(audio_signal * 1.5) * 0.8

    # Add slight pre-emphasis for high frequencies
    pre_emphasis = 0.97
    emphasized = np.zeros_like(compressed)
    emphasized[0] = compressed[0]
    for i in range(1, len(compressed)):
        emphasized[i] = compressed[i] - pre_emphasis * compressed[i - 1]

    return emphasized


def decode_submarine_audio(audio_signal: np.ndarray, tolerance=2) -> str:
    """Decode SUBMARINE mode audio with enhanced robustness"""
    if len(audio_signal) == 0:
        return "❌ No audio data"

    print("🚢 SUBMARINE MODE: Decoding with enhanced robustness")

    # Normalize and filter
    audio_signal = audio_signal.copy().astype(np.float32)
    max_abs = np.max(np.abs(audio_signal))
    if max_abs > 0:
        audio_signal /= max_abs

    # Apply bandpass filter to reduce noise
    nyquist = SAMPLE_RATE / 2
    low_freq = min(CARRIER_FREQS) - 500
    high_freq = max(CARRIER_FREQS) + 500

    if low_freq > 0 and high_freq < nyquist:
        sos = butter(4, [low_freq / nyquist, high_freq / nyquist], btype='band', output='sos')
        audio_signal = filtfilt(sos, audio_signal)

    # Extract symbols
    symbol_len = int(SAMPLE_RATE / SYMBOL_RATE)
    num_chunks = (len(audio_signal) - symbol_len) // symbol_len

    if num_chunks < len(PREAMBLE_SYMBOLS) + len(POSTAMBLE_SYMBOLS):
        return f"❌ Audio too short: {num_chunks} symbols"

    symbols = extract_symbols_robust(audio_signal, symbol_len, num_chunks)
    print(f"📊 Extracted {len(symbols)} symbols")

    # Find all possible message boundaries with tolerance
    preamble_positions = []
    postamble_positions = []

    # Look for multiple preambles/postambles
    search_start = 0
    while search_start < len(symbols) - len(PREAMBLE_SYMBOLS):
        pos = find_sequence_vectorized(symbols[search_start:], PREAMBLE_SYMBOLS, tolerance)
        if pos != -1:
            preamble_positions.append(search_start + pos)
            search_start += pos + len(PREAMBLE_SYMBOLS)
        else:
            break

    search_start = 0
    while search_start < len(symbols) - len(POSTAMBLE_SYMBOLS):
        pos = find_sequence_vectorized(symbols[search_start:], POSTAMBLE_SYMBOLS, tolerance)
        if pos != -1:
            postamble_positions.append(search_start + pos)
            search_start += pos + len(POSTAMBLE_SYMBOLS)
        else:
            break

    print(f"🔍 Found {len(preamble_positions)} preambles, {len(postamble_positions)} postambles")

    if not preamble_positions or not postamble_positions:
        return "❌ Sync patterns not found"

    # Try all combinations of preamble/postamble pairs
    decoded_messages = []

    for pre_pos in preamble_positions:
        for post_pos in postamble_positions:
            if post_pos > pre_pos + len(PREAMBLE_SYMBOLS):
                start_idx = pre_pos + len(PREAMBLE_SYMBOLS)
                end_idx = post_pos

                message_symbols = symbols[start_idx:end_idx]
                if len(message_symbols) > 0:
                    decoded = try_decode_symbols(message_symbols)
                    if decoded and not decoded.startswith("❌"):
                        decoded_messages.append(decoded)
                        print(f"✅ Successfully decoded message variant")

    if decoded_messages:
        # Return the most common decoded message (voting)
        from collections import Counter
        most_common = Counter(decoded_messages).most_common(1)[0][0]
        print(f"🎯 Consensus result from {len(decoded_messages)} successful decodes")
        return most_common

    return "❌ All decode attempts failed"


# ================== SPECIALIZED LIGHTNING MODE ==================
@njit
def generate_lightning_tone(freq, duration, sample_rate, volume):
    """Generate tones optimized for lightning-fast transmission"""
    samples = int(duration * sample_rate)
    t = np.linspace(0, duration, samples).astype(np.float32)

    # Sharp envelope with minimal fade
    envelope = np.ones(samples, dtype=np.float32)
    fade_samples = min(50, samples // 20)  # Very short fade

    for i in range(fade_samples):
        envelope[i] = i / fade_samples
        envelope[samples - 1 - i] = i / fade_samples

    # Generate pure tone with no harmonics for faster processing
    signal = volume * np.sin(2 * np.pi * freq * t)
    return signal * envelope


def encode_lightning_message(message: str) -> np.ndarray:
    """Encode message for LIGHTNING mode with minimal overhead"""
    print(f"⚡ LIGHTNING MODE: Ultra-fast encoding")

    # Minimal compression for speed
    data = zlib.compress(message.encode(), level=1)
    pad_length = (16 - (len(data) % 16)) % 16
    data_padded = data + bytes([0] * pad_length)

    # Apply error correction
    data_encoded = RS_CODEC.encode(data_padded)
    print(f"📦 Data: {len(data)} bytes → {len(data_encoded)} bytes (RS protected)")

    # Convert to symbols
    symbols = bytes_to_symbols_vectorized(np.array(data_encoded, dtype=np.uint8), BITS_PER_SYMBOL)

    # Generate audio
    duration_per_symbol = 1.0 / SYMBOL_RATE
    samples_per_symbol = int(SAMPLE_RATE * duration_per_symbol)
    total_samples = len(symbols) * samples_per_symbol
    audio_signal = np.zeros(total_samples, dtype=np.float32)

    for i in range(len(symbols)):
        symbol = symbols[i]
        freq_idx = symbol % len(CARRIER_FREQS)
        freq = CARRIER_FREQS[freq_idx]

        # Generate tone for this symbol
        start_idx = i * samples_per_symbol
        tone = generate_lightning_tone(freq, duration_per_symbol, SAMPLE_RATE, VOLUME)

        # Add to signal
        audio_signal[start_idx:start_idx + len(tone)] = tone

    # Add minimal pre/postamble
    preamble = generate_lightning_sequence(PREAMBLE_SYMBOLS)
    postamble = generate_lightning_sequence(POSTAMBLE_SYMBOLS)

    silence = np.zeros(int(SAMPLE_RATE * 0.1), dtype=np.float32)
    audio_signal = np.concatenate([silence, preamble, audio_signal, postamble, silence])

    duration = len(audio_signal) / SAMPLE_RATE
    bitrate = len(data_encoded) * 8 / duration
    print(f"⏱️  Duration: {duration:.4f}s, Bitrate: {bitrate / 1000:.1f} kbps")

    return audio_signal


@njit
def generate_lightning_sequence(symbols):
    """Generate optimized sequence for lightning mode"""
    duration_per_symbol = 1.0 / SYMBOL_RATE
    samples_per_symbol = int(SAMPLE_RATE * duration_per_symbol)
    total_samples = len(symbols) * samples_per_symbol
    audio_signal = np.zeros(total_samples, dtype=np.float32)

    for i in range(len(symbols)):
        symbol = symbols[i]
        freq_idx = symbol % len(CARRIER_FREQS)
        freq = CARRIER_FREQS[freq_idx]

        # Generate tone for this symbol
        start_idx = i * samples_per_symbol
        tone = generate_lightning_tone(freq, duration_per_symbol, SAMPLE_RATE, VOLUME)

        # Add to signal
        for j in range(len(tone)):
            audio_signal[start_idx + j] = tone[j]

    return audio_signal


# ================== CORE FUNCTIONS ==================
@njit
def generate_windowed_tones_vectorized(freqs, duration_per_symbol, samples_per_symbol, sample_rate, volume):
    """Vectorized tone generation with Numba JIT optimization"""
    dt = duration_per_symbol / samples_per_symbol
    audio_signal = np.zeros(len(freqs) * samples_per_symbol, dtype=np.float32)

    # Pre-compute Hanning window manually
    window = np.zeros(samples_per_symbol, dtype=np.float32)
    for i in range(samples_per_symbol):
        window[i] = 0.5 - 0.5 * np.cos(2.0 * np.pi * i / (samples_per_symbol - 1))

    # Vectorized generation of all tones
    for i in prange(len(freqs)):
        start_idx = i * samples_per_symbol
        freq = freqs[i]

        # Generate tone with window applied
        for j in range(samples_per_symbol):
            t = j * dt
            phase = 2 * np.pi * freq * t
            audio_signal[start_idx + j] = volume * np.sin(phase) * window[j]

    return audio_signal


def generate_tone_sequence_optimized(symbols):
    """Optimized tone sequence generation"""
    duration_per_symbol = 1.0 / SYMBOL_RATE
    samples_per_symbol = int(SAMPLE_RATE * duration_per_symbol)

    # Convert symbols to frequencies
    symbols_arr = np.array(symbols, dtype=np.int32)
    freqs = CARRIER_FREQS[symbols_arr % len(CARRIER_FREQS)]

    # Generate all tones at once
    audio_signal = generate_windowed_tones_vectorized(
        freqs, duration_per_symbol, samples_per_symbol, SAMPLE_RATE, VOLUME
    )

    return audio_signal


@njit
def symbols_to_bytes_vectorized(symbols, bits_per_symbol):
    """Vectorized conversion of symbols to bytes"""
    data_encoded = np.zeros(len(symbols), dtype=np.uint8)
    byte_count = 0
    current_byte = 0
    bits_collected = 0

    for symbol in symbols:
        current_byte = (current_byte << bits_per_symbol) | symbol
        bits_collected += bits_per_symbol

        if bits_collected >= 8:
            extracted_byte = (current_byte >> (bits_collected - 8)) & 0xFF
            data_encoded[byte_count] = extracted_byte
            byte_count += 1
            bits_collected -= 8
            current_byte &= (1 << bits_collected) - 1

    return data_encoded[:byte_count]


@njit
def bytes_to_symbols_vectorized(data_encoded, bits_per_symbol):
    """Vectorized conversion of bytes to symbols"""
    max_symbols = len(data_encoded) * 8 // bits_per_symbol + 1
    symbols = np.zeros(max_symbols, dtype=np.int32)
    symbol_count = 0

    for byte_val in data_encoded:
        remaining_bits = 8
        while remaining_bits > 0:
            shift = max(0, remaining_bits - bits_per_symbol)
            symbol = (byte_val >> shift) & ((1 << bits_per_symbol) - 1)
            symbols[symbol_count] = symbol
            symbol_count += 1
            remaining_bits -= bits_per_symbol

    return symbols[:symbol_count]


def encode_message_optimized(message: str) -> np.ndarray:
    """Optimized message encoding"""
    # Compression and padding
    data = zlib.compress(message.encode())
    pad_length = (16 - (len(data) % 16)) % 16
    data_padded = data + bytes([0] * pad_length)
    data_encoded = RS_CODEC.encode(data_padded)

    # Vectorized byte to symbol conversion
    symbols = bytes_to_symbols_vectorized(np.array(data_encoded, dtype=np.uint8), BITS_PER_SYMBOL)

    # Generate audio
    preamble_audio = generate_tone_sequence_optimized(PREAMBLE_SYMBOLS)
    message_audio = generate_tone_sequence_optimized(symbols)
    postamble_audio = generate_tone_sequence_optimized(POSTAMBLE_SYMBOLS)

    # Concatenation
    silence_duration = 0.3
    silence = np.zeros(int(SAMPLE_RATE * silence_duration), dtype=np.float32)

    audio_signal = np.concatenate([silence, preamble_audio, message_audio, postamble_audio, silence])
    return audio_signal


@njit
def find_sequence_vectorized(symbols, pattern, tolerance=0):
    """Vectorized sequence finding with JIT optimization"""
    pattern_len = len(pattern)
    symbols_len = len(symbols)

    if tolerance == 0:
        # Exact match
        for i in range(symbols_len - pattern_len + 1):
            match = True
            for j in range(pattern_len):
                if symbols[i + j] != pattern[j]:
                    match = False
                    break
            if match:
                return i
        return -1
    else:
        # Fuzzy match with tolerance
        best_match = -1
        best_score = tolerance + 1

        for i in range(symbols_len - pattern_len + 1):
            diff_count = 0
            for j in range(pattern_len):
                if symbols[i + j] != pattern[j]:
                    diff_count += 1

            if diff_count <= tolerance and diff_count < best_score:
                best_score = diff_count
                best_match = i

        return best_match if best_score <= tolerance else -1


@njit
def extract_symbols_robust(audio_signal, symbol_len, num_chunks):
    """Extract symbols with robust frequency detection"""
    symbols = np.zeros(num_chunks, dtype=np.int32)

    # Pre-compute window
    window = np.zeros(symbol_len, dtype=np.float32)
    for i in range(symbol_len):
        window[i] = 0.5 - 0.5 * np.cos(2.0 * np.pi * i / (symbol_len - 1))

    for i in range(num_chunks):
        start_idx = i * symbol_len
        chunk = audio_signal[start_idx:start_idx + symbol_len] * window

        # Find best matching frequency using correlation
        max_correlation = 0.0
        best_symbol = 0

        for freq_idx in range(len(CARRIER_FREQS)):
            freq = CARRIER_FREQS[freq_idx]
            correlation = 0.0

            # Calculate correlation with reference frequency
            for j in range(len(chunk)):
                t = j / SAMPLE_RATE
                cos_ref = np.cos(2 * np.pi * freq * t)
                sin_ref = np.sin(2 * np.pi * freq * t)
                correlation += chunk[j] * cos_ref + chunk[j] * sin_ref

            correlation = abs(correlation)
            if correlation > max_correlation:
                max_correlation = correlation
                best_symbol = freq_idx

        symbols[i] = best_symbol

    return symbols


def decode_audio_optimized(audio_signal: np.ndarray) -> str:
    """Optimized audio decoding"""
    if len(audio_signal) == 0:
        return "❌ No audio data"

    # Normalize
    audio_signal = audio_signal.copy().astype(np.float32)
    max_abs = np.max(np.abs(audio_signal))
    if max_abs > 0:
        audio_signal /= max_abs

    symbol_len = int(SAMPLE_RATE / SYMBOL_RATE)
    num_chunks = (len(audio_signal) - symbol_len) // symbol_len
    symbols = np.zeros(num_chunks, dtype=np.int32)

    # Create window
    window = np.zeros(symbol_len, dtype=np.float32)
    for i in range(symbol_len):
        window[i] = 0.5 - 0.5 * np.cos(2.0 * np.pi * i / (symbol_len - 1))

    # Pre-compute frequency bins for Real FFT
    freqs = rfftfreq(symbol_len, 1 / SAMPLE_RATE)

    # Process chunks
    for i in range(num_chunks):
        start_idx = i * symbol_len
        chunk = audio_signal[start_idx:start_idx + symbol_len]

        windowed = chunk * window
        fft_result = np.abs(rfft(windowed))
        dominant_bin = np.argmax(fft_result)
        freq_hz = freqs[dominant_bin]

        # Find closest carrier frequency
        symbol = np.argmin(np.abs(CARRIER_FREQS - freq_hz))
        symbols[i] = symbol

    if len(symbols) < len(PREAMBLE_SYMBOLS) + len(POSTAMBLE_SYMBOLS):
        return f"❌ Audio too short: {len(symbols)} symbols"

    # Find sequences
    start_idx = find_sequence_vectorized(symbols, PREAMBLE_SYMBOLS, tolerance=0)
    end_idx = find_sequence_vectorized(symbols, POSTAMBLE_SYMBOLS, tolerance=0)

    if start_idx == -1:
        return f"❌ Preamble not found in {len(symbols)} symbols"
    if end_idx == -1:
        return "❌ Postamble not found"
    if end_idx <= start_idx:
        return "❌ Invalid sync positions"

    message_symbols = symbols[start_idx + len(PREAMBLE_SYMBOLS): end_idx]
    if len(message_symbols) == 0:
        return "❌ No message data"

    # Convert to bytes
    data_encoded = symbols_to_bytes_vectorized(message_symbols, BITS_PER_SYMBOL)

    try:
        data_corrected = RS_CODEC.decode(bytes(data_encoded))[0]
        # Remove padding
        last_non_zero = len(data_corrected)
        while last_non_zero > 0 and data_corrected[last_non_zero - 1] == 0:
            last_non_zero -= 1
        message = zlib.decompress(data_corrected[:last_non_zero]).decode()
        return message
    except Exception as e:
        return f"❌ Decode failed: {str(e)[:50]}"


def try_decode_symbols(message_symbols):
    """Try to decode symbols with error recovery"""
    try:
        data_encoded = symbols_to_bytes_vectorized(message_symbols, BITS_PER_SYMBOL)

        if len(data_encoded) < RS_CODEC.nsym:
            return "❌ Insufficient data for Reed-Solomon"

        # Try Reed-Solomon decode
        data_corrected = RS_CODEC.decode(bytes(data_encoded))[0]

        # Remove padding
        last_non_zero = len(data_corrected)
        while last_non_zero > 0 and data_corrected[last_non_zero - 1] == 0:
            last_non_zero -= 1

        # Decompress
        message = zlib.decompress(data_corrected[:last_non_zero]).decode()
        return message

    except Exception as e:
        return f"❌ Decode error: {str(e)[:30]}"


# ================== NOISE GENERATION ==================
@njit
def generate_pink_noise(length, sample_rate):
    """Generate pink noise (1/f noise) using Numba JIT"""
    white = np.random.normal(0, 1, length).astype(np.float32)

    # Pink noise filter approximation
    filtered = np.zeros(length, dtype=np.float32)
    b = np.array([0.049922035, -0.095993537, 0.050612699, -0.004408786], dtype=np.float32)
    a = np.array([1, -2.494956002, 2.017265875, -0.522189400], dtype=np.float32)

    # Simple IIR filter
    for i in range(length):
        filtered[i] = b[0] * white[i]
        for j in range(1, min(4, i + 1)):
            if i - j >= 0:
                filtered[i] += b[j] * white[i - j] - a[j] * filtered[i - j]

    return filtered


@njit
def generate_brown_noise(length):
    """Generate brown noise (Brownian/red noise) using Numba JIT"""
    white = np.random.normal(0, 1, length).astype(np.float32)
    brown = np.zeros(length, dtype=np.float32)

    # Brown noise is integrated white noise
    brown[0] = white[0]
    for i in range(1, length):
        brown[i] = brown[i - 1] + white[i] * 0.02

    return brown


def add_noise_to_audio(audio_signal, noise_type="white", snr_db=20):
    """Add different types of noise to audio signal at specified SNR"""
    if len(audio_signal) == 0:
        return audio_signal

    # Calculate signal power
    signal_power = np.mean(audio_signal ** 2)

    # Generate noise based on type
    if noise_type == "white":
        noise = np.random.normal(0, 1, len(audio_signal)).astype(np.float32)
    elif noise_type == "pink":
        noise = generate_pink_noise(len(audio_signal), SAMPLE_RATE)
    elif noise_type == "brown":
        noise = generate_brown_noise(len(audio_signal))
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")

    # Normalize noise
    noise = noise / np.sqrt(np.mean(noise ** 2))

    # Calculate noise power for desired SNR
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = noise * np.sqrt(noise_power)

    return audio_signal + noise


# ================== TESTING & EVALUATION ==================
def submarine_test():
    """Test SUBMARINE mode with various noise conditions"""
    print("🚢" * 20)
    print("SUBMARINE MODE - WALL PENETRATION TEST")
    print("🚢" * 20)

    # Set mode parameters first so config is up-to-date
    setup_mode_params("SUBMARINE")

    test_message = "SUBMARINE MODE: This message should penetrate walls and obstacles!"

    print(f"📝 Test message: '{test_message}'")
    print(f"📊 Configuration:")
    print(f"   • Carriers: {len(CARRIER_FREQS)} ({min(CARRIER_FREQS):.0f}-{max(CARRIER_FREQS):.0f} Hz)")
    print(f"   • Symbol rate: {SYMBOL_RATE} symbols/sec")
    print(f"   • Bits per symbol: {BITS_PER_SYMBOL}")
    print(f"   • Reed-Solomon: {RS_CODEC.nsym} redundancy bytes")
    print()

    # Encode with triple redundancy
    start_time = time.perf_counter()
    audio = encode_submarine_message(test_message, repeat_count=3)
    encode_time = (time.perf_counter() - start_time) * 1000
    print(f"⚡ Encoding time: {encode_time:.2f}ms")

    # Test different noise levels
    snr_levels = [30, 20, 15, 10, 5, 0, -5, -10]
    noise_types = ["white", "pink", "brown"]

    print(f"\n🔊 NOISE ROBUSTNESS TEST")
    print("-" * 60)
    print(f"{'Noise Type':<12} ", end="")
    for snr in snr_levels:
        print(f"{snr:>6}dB ", end="")
    print()
    print("-" * 60)

    for noise_type in noise_types:
        print(f"{noise_type.upper():<12} ", end="")
        for snr in snr_levels:
            try:
                noisy_audio = add_noise_to_audio(audio, noise_type, snr)
                start_time = time.perf_counter()
                decoded = decode_submarine_audio(noisy_audio, tolerance=2)
                decode_time = (time.perf_counter() - start_time) * 1000
                success = decoded == test_message
                symbol = "✅" if success else "❌"
                print(f"{symbol:>6}  ", end="")
            except Exception:
                print(f"{'E':>6}  ", end="")
        print()

    print()
    print("🎯 CLEAN SIGNAL TEST")
    decoded_clean = decode_submarine_audio(audio, tolerance=0)
    print(f"Result: {decoded_clean}")
    print(f"Success: {'✅' if decoded_clean == test_message else '❌'}")



def lightning_test():
    """Test LIGHTNING mode performance"""
    print("⚡" * 20)
    print("LIGHTNING MODE - SPEED TEST")
    print("⚡" * 20)

    test_message = "LIGHTNING MODE: Ultra-fast transmission at 800 symbols/sec!"

    print(f"📝 Test message: '{test_message}'")
    print(f"📊 Configuration:")
    print(f"   • Carriers: {len(CARRIER_FREQS)} ({min(CARRIER_FREQS):.0f}-{max(CARRIER_FREQS):.0f} Hz)")
    print(f"   • Symbol rate: {SYMBOL_RATE} symbols/sec")
    print(f"   • Bits per symbol: {BITS_PER_SYMBOL}")
    print(f"   • Reed-Solomon: {RS_CODEC.nsym} redundancy bytes")
    print()

    # Set mode parameters
    setup_mode_params("LIGHTNING")

    # Encode
    start_time = time.perf_counter()
    audio = encode_lightning_message(test_message)
    encode_time = (time.perf_counter() - start_time) * 1000

    print(f"⚡ Encoding time: {encode_time:.2f}ms")

    # Test clean signal
    print("\n🎯 CLEAN SIGNAL TEST")
    start_time = time.perf_counter()
    decoded = decode_audio_optimized(audio)
    decode_time = (time.perf_counter() - start_time) * 1000

    print(f"Result: {decoded}")
    print(f"Success: {'✅' if decoded == test_message else '❌'}")
    print(f"Decode time: {decode_time:.2f}ms")

    # Test with noise
    print("\n🔊 NOISE ROBUSTNESS TEST")
    for snr in [20, 15, 10, 5]:
        noisy_audio = add_noise_to_audio(audio, "white", snr)
        decoded = decode_audio_optimized(noisy_audio)
        success = decoded == test_message
        print(f"SNR: {snr}dB - {'✅' if success else '❌'} - {decoded[:50]}")


def comprehensive_test():
    """Comprehensive test with bitrates and noise robustness"""
    test_message = "The quick brown fox jumps over the lazy dog" * 5
    modes = ["FAST", "FASTEST", "SUBMARINE", "ULTRASOUND", "LIGHTNING"]
    noise_types = ["white", "pink", "brown"]
    snr_levels = [30, 20, 15, 10, 5, 0, -5]  # dB

    print("🚀 COMPREHENSIVE AUDIO MODEM TEST")
    print("=" * 80)

    # First, show clean performance and bitrates
    print("\n📊 CLEAN SIGNAL PERFORMANCE & BITRATES")
    print("-" * 80)
    print(f"{'Mode':<15} {'Bitrate':<12} {'Encode(ms)':<12} {'Decode(ms)':<12} {'Efficiency':<12}")
    print("-" * 80)

    mode_results = {}

    for mode in modes:
        try:
            setup_mode_params(mode)

            # Calculate theoretical and effective bitrates
            theoretical_bps = SYMBOL_RATE * BITS_PER_SYMBOL
            theoretical_kbps = theoretical_bps / 1000

            # Calculate effective bitrate (accounting for overhead)
            data = zlib.compress(test_message.encode())
            pad_length = (16 - (len(data) % 16)) % 16
            data_padded = data + bytes([0] * pad_length)
            data_encoded = RS_CODEC.encode(data_padded)
            overhead_ratio = len(data) / len(data_encoded)
            effective_kbps = theoretical_kbps * overhead_ratio

            # Performance test
            start_time = time.perf_counter()
            if mode == "SUBMARINE":
                audio = encode_submarine_message(test_message, repeat_count=3)
            elif mode == "LIGHTNING":
                audio = encode_lightning_message(test_message)
            else:
                audio = encode_message_optimized(test_message)
            encode_time = (time.perf_counter() - start_time) * 1000  # ms

            start_time = time.perf_counter()
            if mode == "SUBMARINE":
                decoded = decode_submarine_audio(audio)
            else:
                decoded = decode_audio_optimized(audio)
            decode_time = (time.perf_counter() - start_time) * 1000  # ms

            success = decoded == test_message
            efficiency = "✓" if success else "✗"

            mode_results[mode] = {
                'audio': audio,
                'theoretical_kbps': theoretical_kbps,
                'effective_kbps': effective_kbps,
                'encode_time': encode_time,
                'decode_time': decode_time
            }

            print(f"{mode:<15} {effective_kbps:<8.2f}kbps {encode_time:<8.2f}ms {decode_time:<8.2f}ms {efficiency:<12}")

        except Exception as e:
            print(f"{mode:<15} ERROR: {str(e)[:40]}")
            continue

    # Noise robustness testing
    print(f"\n🔊 NOISE ROBUSTNESS TEST")
    print("-" * 80)
    print(f"Testing modes: {', '.join(mode_results.keys())}")
    print(f"Noise types: {', '.join(noise_types)}")
    print(f"SNR levels: {snr_levels} dB")
    print()

    for noise_type in noise_types:
        print(f"\n📡 {noise_type.upper()} NOISE RESULTS")
        print("-" * 60)
        print(f"{'Mode':<15} ", end="")
        for snr in snr_levels:
            print(f"{snr:>6}dB ", end="")
        print()
        print("-" * 60)

        for mode, results in mode_results.items():
            setup_mode_params(mode)  # Reset mode params
            print(f"{mode:<15} ", end="")

            for snr in snr_levels:
                try:
                    # Add noise to the clean audio
                    noisy_audio = add_noise_to_audio(results['audio'], noise_type, snr)

                    # Try to decode
                    if mode == "SUBMARINE":
                        decoded = decode_submarine_audio(noisy_audio)
                    else:
                        decoded = decode_audio_optimized(noisy_audio)
                    success = decoded == test_message

                    symbol = "✓" if success else "✗"
                    print(f"{symbol:>6}  ", end="")

                except Exception:
                    print(f"{'E':>6}  ", end="")

            print()  # New line after each mode

        print()  # Extra space between noise types

    # Summary statistics
    print("\n📈 SUMMARY STATISTICS")
    print("-" * 40)
    for mode, results in mode_results.items():
        print(f"{mode}:")
        print(f"  Theoretical: {results['theoretical_kbps']:.2f} kbps")
        print(f"  Effective:   {results['effective_kbps']:.2f} kbps")
        print(
            f"  Overhead:    {((results['theoretical_kbps'] - results['effective_kbps']) / results['theoretical_kbps'] * 100):.1f}%")
        print(f"  Encode:      {results['encode_time']:.2f} ms")
        print(f"  Decode:      {results['decode_time']:.2f} ms")
        print()


# ================== MAIN EXECUTION ==================
if __name__ == "__main__":
    # Run specialized tests
    print("=" * 80)
    submarine_test()

    print("\n" + "=" * 80)
    lightning_test()

    # Run comprehensive test
    print("\n" + "=" * 80)
    comprehensive_test()

    print("\n" + "=" * 80)
    print("🚢 SUBMARINE MODE FEATURES:")
    print("=" * 80)
    print("✅ Triple redundancy transmission")
    print("✅ Extended preamble/postamble patterns")
    print("✅ 32 carrier frequencies (2-24 kHz)")
    print("✅ Strong Reed-Solomon error correction (80 bytes)")
    print("✅ Fuzzy pattern matching with tolerance")
    print("✅ Harmonic enhancement for penetration")
    print("✅ Bandpass filtering for noise reduction")
    print("✅ Consensus decoding from multiple attempts")
    print("✅ Pre-emphasis for high-frequency boost")
    print("✅ Enhanced envelope shaping")
    print("\n🎯 Optimized for: Wall penetration, obstacle bypass, extreme noise")

    print("\n" + "=" * 80)
    print("⚡ LIGHTNING MODE FEATURES:")
    print("=" * 80)
    print("✅ Ultra-fast 800 symbols/sec transmission")
    print("✅ Wide frequency range (2-28 kHz)")
    print("✅ Minimal preamble/postamble overhead")
    print("✅ Sharp envelope transitions")
    print("✅ Optimized for clean line-of-sight conditions")
    print("✅ Efficient processing pipeline")
    print("\n🎯 Optimized for: High-speed data transfer, short-range communication")

    # Show bitrate info
    print("\n🔬 Noise Types:")
    print(f"• White Noise: Uniform across all frequencies")
    print(f"• Pink Noise:  1/f spectrum (natural background)")
    print(f"• Brown Noise: 1/f² spectrum (deeper rumble)")
    print(f"\n📡 SNR Levels: 30dB (clean) → -10dB (extreme noise)")