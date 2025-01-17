# ziwen chen
# zchen56
from venv import create

import imageio
import numpy as np
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from skimage import measure  # 用于提取等值面


# 获得数据
file_path = "HW1 DATA/subdata.npy"
data = np.load(file_path)
print("数据形状: ", data.shape)
# print("数据类型: ", data.dtype)
# print("数据的平均值: ", np.mean(data))
# print("数据的最大值: ", np.max(data))
# print("数据的最小值: ", np.min(data))


'''
# ------------------------------------------------------------------------------
#                                   《Task 1》
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# Step 0: Parameters / Domain Setup
# ------------------------------------------------------------------------------
# Suppose your physical domain is x, y, z in [-L, L].
L = 10.0

# ------------------------------------------------------------------------------
# Step 1: Load and Reshape the Submarine Data
# ------------------------------------------------------------------------------
data_reshaped = data.reshape((64, 64, 64, 49))
nx, ny, nz, nt = data_reshaped.shape  # Expect (64, 64, 64, 49)

# ------------------------------------------------------------------------------
# Step 2: Compute the Averaged 3D Fourier Transform
# ------------------------------------------------------------------------------
data_fft = np.fft.fftn(data_reshaped, axes = (0, 1, 2)) # 对 “xyz” 进行分析，以t为轴。
# [错误：我这里犯了一个错误，我一开始把时间也进行了傅立叶变换，但是时间在这里只做参考意义，没有任何周期性的变化所以不需要进行转换]
# [修正：仅将时间作为参考轴，单纯分析xzy。对xyz进行 “切片” 式的傅立叶转换，我会得到 （kx，ky，kt，t），k为频域，t为正常的时间]

mean_data_fft = np.mean(data_fft, axis=3) # 以时间轴，将所有频域平均，过滤掉噪音。
# [将每个时间段 t_n 的 频域(kx, ky, kz) 相加除以总时间 nt，获得每个频域的平均 (kx_avg，ky_avg, kz_avg)]


# ------------------------------------------------------------------------------
# Step 3: Shift the Zero Frequency to the Center and Find the Magnitude
# ------------------------------------------------------------------------------
ft_avg_shifted = np.fft.fftshift(mean_data_fft) # 中心化
mag_ft_avg = np.abs(ft_avg_shifted) # 求magnitude

# ------------------------------------------------------------------------------
# Step 4: Identify the Dominant Frequency (Index of Maximum in 3D FFT)
# ------------------------------------------------------------------------------
max_idx = np.unravel_index(np.argmax(mag_ft_avg), mag_ft_avg.shape)
print("Index of dominant frequency in shifted coords:", max_idx)

# ------------------------------------------------------------------------------
# Step 5: Construct the Frequency Axes
# ------------------------------------------------------------------------------
kx_shifted = ky_shifted = kz_shifted = np.fft.fftshift(np.fft.fftfreq(nx))

kx_dom = kx_shifted[max_idx[0]]
ky_dom = ky_shifted[max_idx[1]]
kz_dom = kz_shifted[max_idx[2]]
print(f"Dominant freq (normalized) = ({kx_dom}, {ky_dom}, {kz_dom})")

# ------------------------------------------------------------------------------
# Step 6: Convert to Physical Wave Numbers (Assume x,y,z in [-L, L])
# ------------------------------------------------------------------------------
kx_dom_phys = (kx_dom * nx) * (np.pi / L)
ky_dom_phys = (ky_dom * ny) * (np.pi / L)
kz_dom_phys = (kz_dom * nz) * (np.pi / L)

print("Dominant freq in physical wave numbers (assuming domain [-L,L]):")
print(f"({kx_dom_phys:.3f}, {ky_dom_phys:.3f}, {kz_dom_phys:.3f})")

# ------------------------------------------------------------------------------
# Step 7: Amplitude at the Dominant Frequency
# ------------------------------------------------------------------------------
dominant_freq_amplitude = mag_ft_avg[max_idx]
print(f"Amplitude at dominant frequency: {dominant_freq_amplitude:.3f}")

# ------------------------------------------------------------------------------
# Step 8: 2D Slice Visualization at the Dominant k_z
# ------------------------------------------------------------------------------
dominant_kz_idx = max_idx[2]  # z-slice index of the dominant freq
extent_xy = [kx_shifted[0], kx_shifted[-1], ky_shifted[0], ky_shifted[-1]]

plt.figure(figsize=(6, 5))
# Transpose so that x-axis is horizontal, y-axis is vertical in usual orientation
plt.imshow(mag_ft_avg[:, :, dominant_kz_idx].T,
           extent=extent_xy, origin='lower', aspect='auto')
plt.colorbar(label="|FFT| amplitude")
plt.title(f"Slice at kz index = {dominant_kz_idx}")

# Mark the dominant frequency in this slice with a red star
plt.plot(kx_dom, ky_dom, 'r*', markersize=12, label="Dominant Frequency")

plt.xlabel("kx")
plt.ylabel("ky")
plt.legend()
plt.show()

# ------------------------------------------------------------------------------
# Step 9: 3D Isosurface Visualization with Plotly
# ------------------------------------------------------------------------------
# 9a) Create a 3D meshgrid of (kx, ky, kz) in shifted frequency space
kxx, kyy, kzz = np.meshgrid(kx_shifted, ky_shifted, kz_shifted, indexing='ij')

# Flatten the arrays for Plotly's isosurface needs
kxx_flat = kxx.flatten()
kyy_flat = kyy.flatten()
kzz_flat = kzz.flatten()
mag_flat = mag_ft_avg.flatten()

# 9b) Plot an isosurface around ~70% of the max amplitude
iso_min = 0.7 * np.max(mag_ft_avg)
iso_max = np.max(mag_ft_avg)

fig = go.Figure(data=go.Isosurface(
    x=kxx_flat,
    y=kyy_flat,
    z=kzz_flat,
    value=mag_flat,
    isomin=iso_min,
    isomax=iso_max,
    surface_count=3,
    caps=dict(x_show=False, y_show=False, z_show=False),
    colorscale='Viridis',
    showscale=True
))

# 9c) Mark the dominant frequency with a 3D scatter point
fig.add_trace(go.Scatter3d(
    x=[kx_dom],
    y=[ky_dom],
    z=[kz_dom],
    mode='markers',
    marker=dict(size=5, color='red', symbol='x'),
    name='Dominant Freq'
))

fig.update_layout(
    title="3D Isosurface of FFT Magnitude (Averaged)",
    scene=dict(
        xaxis_title="kx (shifted)",
        yaxis_title="ky (shifted)",
        zaxis_title="kz (shifted)"
    ),
    width=800,
    height=700
)

fig.show()
'''

# ------------------------------------------------------------------------------
#                                   《Task 2》
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# Data filter function
# ------------------------------------------------------------------------------
def fft_3d_filter (data, cutoff_frequency):
    """
    Apply a 3D low-pass FFT filter to a 4D array (n, n, n, t).

    Parameters:
    - data: np.ndarray
        Input 4D array with shape (n, n, n, t).
    - cutoff_frequency: float
        The cutoff frequency for the low-pass filter. Frequencies higher than this will be suppressed.

    Returns:
    - filtered_data: np.ndarray
        The filtered 4D array with the same shape as the input data.
    """
    # Check input dimensions
    if data.ndim != 4:
        raise ValueError ("Input data must be a 4D array with shape (n, n, n, t).")
    if cutoff_frequency < 0 or cutoff_frequency >= 1:
        raise ValueError ("Cutoff frequency must be [0,1)")

    n, _, _, t = data.shape

    # Initialize the output array
    filtered_data = np.zeros_like (data, dtype=np.float64)

    # Create frequency grid for filtering, 每个frequency的范围 是 [-0.5, 0.5)
    freq_x = np.fft.fftfreq (n)
    freq_y = np.fft.fftfreq (n)
    freq_z = np.fft.fftfreq (n)
    freq_x, freq_y, freq_z = np.meshgrid (freq_x, freq_y, freq_z, indexing='ij')
    freq_magnitude = np.sqrt (freq_x ** 2 + freq_y ** 2 + freq_z ** 2)

    # Create a low-pass filter mask
    filter_mask = freq_magnitude <= cutoff_frequency

    # Apply the filter to each time slice
    for t_idx in range (t):
        # Perform 3D FFT
        fft_data = np.fft.fftn (data[..., t_idx])

        # Apply the filter in the frequency domain
        fft_data_filtered = fft_data * filter_mask

        # Perform inverse FFT to transform back to the spatial domain
        filtered_data[..., t_idx] = np.fft.ifftn (fft_data_filtered).real

    return filtered_data


def process_frequency_data(data, cutoff_frequency):
    """
    先将数据转换到频域，先进行高频过滤（低通滤波），然后再对时间维度平均以提取主频率。

    Parameters:
    - data: np.ndarray
        输入 4D 数据，形状为 (n, n, n, t)。
    - cutoff_frequency: float
        低通滤波器的截止频率，范围为 [0, 0.5)。

    Returns:
    - filtered_data: np.ndarray
        空间域的过滤后数据。
    - dominant_frequency: tuple
        主频率的索引 (fx, fy, fz)。
    """
    # 检查输入数据
    if data.ndim != 4:
        raise ValueError("输入数据必须是 4D 数组，形状为 (n, n, n, t)。")

    n_x, n_y, n_z, n_t = data.shape

    # Step 1: 转换到频域
    fft_data = np.fft.fftn(data, axes=(0, 1, 2))  # 对空间轴做傅里叶变换

    # Step 2: 创建频率网格
    freq_x = np.fft.fftfreq(n_x)
    freq_y = np.fft.fftfreq(n_y)
    freq_z = np.fft.fftfreq(n_z)
    freq_x, freq_y, freq_z = np.meshgrid(freq_x, freq_y, freq_z, indexing='ij')
    freq_magnitude = np.sqrt(freq_x**2 + freq_y**2 + freq_z**2)

    # Step 3: 高频过滤（低通滤波）
    filter_mask = freq_magnitude <= cutoff_frequency
    fft_data_filtered = fft_data * filter_mask[..., np.newaxis]  # 应用到所有时间切片

    # Step 4: 时间维度平均
    fft_avg = np.mean(np.abs(fft_data_filtered), axis=3)  # 对时间轴取平均，得到 3D 频谱

    # Step 5: 提取主频率
    dominant_index = np.unravel_index(np.argmax(fft_avg), fft_avg.shape)
    dominant_frequency = (freq_x[dominant_index], freq_y[dominant_index], freq_z[dominant_index])

    # Step 6: 回到空间域
    filtered_data = np.fft.ifftn(fft_data_filtered, axes=(0, 1, 2)).real  # 转换回空间域

    return filtered_data, dominant_frequency


# ------------------------------------------------------------------------------
# Drawing isoSurface gif
# ------------------------------------------------------------------------------
def create_isosurface_gif(data, isovalue, gif_filename, fps=10):
    """
    Create an isosurface animation (GIF) from 4D data.

    Parameters:
    - data: np.ndarray
        A 4D numpy array with shape (64, 64, 64, 49), representing (x, y, z, t).
    - isovalue: float
        The value at which to generate the isosurface.
    - gif_filename: str
        The filename for the output GIF.
    - fps: int
        Frames per second for the GIF.
    """
    # Check input dimensions
    if data.ndim != 4:
        raise ValueError("Input data must be a 4D array with shape (64, 64, 64, t).")

    n_x, n_y, n_z, n_t = data.shape

    # Create a meshgrid for spatial dimensions
    x = np.linspace(0, n_x - 1, n_x)
    y = np.linspace(0, n_y - 1, n_y)
    z = np.linspace(0, n_z - 1, n_z)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # Create a list to store frames for the GIF
    frames = []

    # Loop through time slices
    for t in range(n_t):
        fig = plt.figure(figsize=(8, 8))  # 调整宽高以匹配实际分辨率
        ax = fig.add_subplot(111, projection='3d')

        # Extract the 3D data for the current time slice
        data_t = data[..., t]

        # 检查当前数据范围是否包含 isovalue
        if np.min (data_t) > isovalue or np.max (data_t) < isovalue:
            print (f"Skipping time slice {t}: No data within isovalue range.")
            plt.close (fig)  # 显式关闭图形对象
            continue  # 跳过当前时间点

        try:
            # Extract the isosurface using skimage.measure.marching_cubes
            verts, faces, _, _ = measure.marching_cubes (data_t, level=isovalue)
        except ValueError as e:
            print (f"Skipping time slice {t}: {e}")
            plt.close (fig)  # 显式关闭图形对象
            continue  # 跳过当前时间点

        # Plot the isosurface
        ax.plot_trisurf(
            verts[:, 0], verts[:, 1], verts[:, 2],
            triangles=faces, cmap='viridis', alpha=0.8
        )

        # Set plot limits and labels
        ax.set_xlim(0, n_x)
        ax.set_ylim(0, n_y)
        ax.set_zlim(0, n_z)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(f"Time: {t + 1}/{n_t}")

        # Save the current frame as an image
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_argb(), dtype='uint8')  # 使用 ARGB
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (4,))  # 4 通道
        frame = frame[:, :, 1:]  # 转换为 RGB
        frames.append(frame)

        # Close the figure to free memory
        plt.close(fig)

    # Save all frames as a GIF
    imageio.mimsave(gif_filename, frames, fps=fps)
    print(f"GIF saved as {gif_filename}")


# filtered_data = fft_3d_filter(data.reshape(64, 64, 64, 49), 0.5)
filtered_data, dominant_frequency = process_frequency_data(data.reshape(64,64,64,49), 0.5)
print(dominant_frequency)
create_isosurface_gif(filtered_data, isovalue=0.5, gif_filename="filtered_Data.gif")
create_isosurface_gif(data.reshape(64, 64, 64, 49), isovalue=0.5, gif_filename="original_Data.gif")




