import sys
import os
import cv2
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QTabWidget, 
                             QFileDialog, QSlider, QGroupBox, QFormLayout, 
                             QScrollArea, QSpinBox, QComboBox, QCheckBox, 
                             QMessageBox, QRadioButton, QButtonGroup, QLineEdit, QGridLayout)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPixmap, QImage, QAction, QIcon
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from scipy import ndimage

# --- UTILS & CUSTOM WIDGETS ---

class ImageLabel(QLabel):
    """Custom Label to handle mouse events for pixel info (Week 1)"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.main_window = None  # Reference to main window to update status bar
        self.original_image_data = None # Store the numpy array

    def set_main_window(self, window):
        self.main_window = window

    def set_image_data(self, image):
        self.original_image_data = image

    def mouseMoveEvent(self, event):
        if self.original_image_data is not None and self.main_window:
            # Calculate coordinates based on scaling
            pixmap = self.pixmap()
            if not pixmap: return
            
            # Simple mapping if no scaling (or strictly handled scaling)
            # For simplicity in this assignment, assuming direct mapping or 1:1 view for pixel peeping
            # Ideally, we calculate ratio between widget size and pixmap size
            
            x = event.pos().x()
            y = event.pos().y()
            
            h, w = self.original_image_data.shape[:2]
            
            # Clamp coordinates
            if 0 <= x < w and 0 <= y < h:
                if len(self.original_image_data.shape) == 3:
                    val = self.original_image_data[y, x] # BGR
                    val_str = f"R:{val[2]} G:{val[1]} B:{val[0]}"
                else:
                    val = self.original_image_data[y, x]
                    val_str = f"Gray:{val}"
                self.main_window.status_label.setText(f"Coords: ({x}, {y}) | Intensity: {val_str}")
            else:
                self.main_window.status_label.setText("Out of bounds")
        super().mouseMoveEvent(event)

class HistogramCanvas(FigureCanvas):
    """Matplotlib Canvas for Week 3"""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig, self.axes = plt.subplots(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)

# --- MAIN APPLICATION ---

class CVApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Computer Vision Course Project (Week 1-8)")
        self.setGeometry(100, 100, 1280, 800)

        # Main Data
        self.original_image = None
        self.processed_image = None
        self.current_week_idx = 0

        # UI Setup
        self.init_ui()

    def init_ui(self):
        # Central Widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Top Bar: Load Image
        top_layout = QHBoxLayout()
        self.btn_load = QPushButton("📂 Load Image")
        self.btn_load.clicked.connect(self.load_image)
        self.lbl_image_info = QLabel("No image loaded.")
        top_layout.addWidget(self.btn_load)
        top_layout.addWidget(self.lbl_image_info)
        top_layout.addStretch()
        main_layout.addLayout(top_layout)

        # Tabs for Weeks
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        self.tab_w1 = QWidget()
        self.tab_w2 = QWidget()
        self.tab_w3 = QWidget()
        self.tab_w4 = QWidget()
        self.tab_w5 = QWidget()
        self.tab_w6 = QWidget()
        self.tab_w7 = QWidget()
        self.tab_w8 = QWidget()

        self.tabs.addTab(self.tab_w1, "Week 1: Basics")
        self.tabs.addTab(self.tab_w2, "Week 2: Point Proc")
        self.tabs.addTab(self.tab_w3, "Week 3: Histogram")
        self.tabs.addTab(self.tab_w4, "Week 4: Spatial")
        self.tabs.addTab(self.tab_w5, "Week 5: Frequency")
        self.tabs.addTab(self.tab_w6, "Week 6: PCA/Face")
        self.tabs.addTab(self.tab_w7, "Week 7: Restoration")
        self.tabs.addTab(self.tab_w8, "Week 8: Morphology")

        # Setup Content for each tab
        self.setup_week1()
        self.setup_week2()
        self.setup_week3()
        self.setup_week4()
        self.setup_week5()
        self.setup_week6()
        self.setup_week7()
        self.setup_week8()

        # Shared Image Viewer Area (Used by most tabs, but some have custom views)
        # We will use a flexible approach: Each tab has specific controls, 
        # but results often update a common display or tab-specific display.
        # For simplicity in this monolithic app, we use Tab-Specific Displays.

        # Status Bar
        self.status_label = QLabel("Ready")
        self.statusBar().addWidget(self.status_label)

    # --- COMMON FUNCTIONS ---
    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.bmp *.tif)")
        if file_name:
            self.original_image = cv2.imread(file_name)
            # Convert to RGB mostly, keep Grayscale copy if needed
            self.update_image_info()
            self.reset_tabs_data()
            self.show_image_week1() # Default show

    def update_image_info(self):
        if self.original_image is not None:
            h, w = self.original_image.shape[:2]
            self.lbl_image_info.setText(f"Size: {w}x{h} | Origin: Top-Left (0,0)")

    def cv_to_qt_pixmap(self, cv_img):
        """Convert OpenCV image to QPixmap"""
        if cv_img is None: return QPixmap()
        
        # Handle grayscale
        if len(cv_img.shape) == 2:
            h, w = cv_img.shape
            bytes_per_line = w
            q_img = QImage(cv_img.data, w, h, bytes_per_line, QImage.Format.Format_Grayscale8)
        else:
            # Handle RGB
            h, w, ch = cv_img.shape
            bytes_per_line = ch * w
            rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            q_img = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            
        return QPixmap.fromImage(q_img)

    def reset_tabs_data(self):
        # Refresh displays on current tab
        idx = self.tabs.currentIndex()
        if idx == 0: self.show_image_week1()
        # Add logic for other tabs if they need auto-refresh on load

    # ================= WEEK 1: Basics =================
    def setup_week1(self):
        layout = QHBoxLayout(self.tab_w1)
        
        # Controls
        controls = QGroupBox("Controls")
        ctrl_layout = QVBoxLayout()
        
        # Spatial Resolution
        ctrl_layout.addWidget(QLabel("Spatial Resolution (Sampling):"))
        self.w1_slider_spatial = QSlider(Qt.Orientation.Horizontal)
        self.w1_slider_spatial.setRange(32, 1024) # Min 32 pixels
        self.w1_slider_spatial.setValue(512)
        self.w1_slider_spatial.valueChanged.connect(self.w1_process)
        ctrl_layout.addWidget(self.w1_slider_spatial)

        # Intensity Resolution
        ctrl_layout.addWidget(QLabel("Intensity Resolution (Bits):"))
        self.w1_combo_bits = QComboBox()
        self.w1_combo_bits.addItems(["8-bit", "4-bit", "2-bit", "1-bit"])
        self.w1_combo_bits.currentIndexChanged.connect(self.w1_process)
        ctrl_layout.addWidget(self.w1_combo_bits)
        
        ctrl_layout.addStretch()
        controls.setLayout(ctrl_layout)
        
        # Display
        self.w1_display = ImageLabel()
        self.w1_display.set_main_window(self)
        self.w1_display.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        
        scroll = QScrollArea()
        scroll.setWidget(self.w1_display)
        scroll.setWidgetResizable(True)

        layout.addWidget(controls, 1)
        layout.addWidget(scroll, 3)

    def show_image_week1(self):
        if self.original_image is not None:
            self.w1_process()

    def w1_process(self):
        if self.original_image is None: return
        
        # 1. Spatial Resolution (Resize)
        target_w = self.w1_slider_spatial.value()
        h, w = self.original_image.shape[:2]
        ratio = h / w
        target_h = int(target_w * ratio)
        
        # Downsample then Upsample to simulate pixelation effect while keeping display size viewable
        small = cv2.resize(self.original_image, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        res_img = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

        # 2. Intensity Resolution (Quantization)
        bits_txt = self.w1_combo_bits.currentText()
        bits = int(bits_txt.split('-')[0])
        
        levels = 2 ** bits
        # Quantize: floor(val / interval) * interval + mid_interval
        interval = 256 / levels
        quantized = np.floor(res_img / interval) * interval
        quantized = quantized.astype(np.uint8)

        self.w1_display.setPixmap(self.cv_to_qt_pixmap(quantized))
        self.w1_display.set_image_data(quantized)

    # ================= WEEK 2: Point Processing =================
    def setup_week2(self):
        layout = QHBoxLayout(self.tab_w2)
        
        # Controls
        ctrl_panel = QGroupBox("Point Processing")
        form = QVBoxLayout()

        # Invert
        btn_invert = QPushButton("Invert Color (Negative)")
        btn_invert.clicked.connect(self.w2_invert)
        form.addWidget(btn_invert)

        # Threshold
        form.addWidget(QLabel("Threshold (0-255):"))
        self.w2_slider_thresh = QSlider(Qt.Orientation.Horizontal)
        self.w2_slider_thresh.setRange(0, 255)
        self.w2_slider_thresh.sliderReleased.connect(self.w2_threshold)
        form.addWidget(self.w2_slider_thresh)

        # Log
        btn_log = QPushButton("Apply Log Transform")
        btn_log.clicked.connect(self.w2_log)
        form.addWidget(btn_log)

        # Gamma
        form.addWidget(QLabel("Gamma (0.1 - 5.0):"))
        self.w2_slider_gamma = QSlider(Qt.Orientation.Horizontal)
        self.w2_slider_gamma.setRange(1, 50) # Divide by 10
        self.w2_slider_gamma.setValue(10)
        self.w2_slider_gamma.sliderReleased.connect(self.w2_gamma)
        form.addWidget(self.w2_slider_gamma)

        # Slicing
        form.addWidget(QLabel("Gray-level Slicing (Min-Max):"))
        slice_layout = QHBoxLayout()
        self.w2_spin_min = QSpinBox(); self.w2_spin_min.setRange(0, 255); self.w2_spin_min.setValue(100)
        self.w2_spin_max = QSpinBox(); self.w2_spin_max.setRange(0, 255); self.w2_spin_max.setValue(200)
        slice_layout.addWidget(self.w2_spin_min); slice_layout.addWidget(self.w2_spin_max)
        form.addLayout(slice_layout)
        self.w2_radio_bg = QRadioButton("Keep BG"); self.w2_radio_bg.setChecked(True)
        self.w2_radio_black = QRadioButton("Black BG")
        form.addWidget(self.w2_radio_bg); form.addWidget(self.w2_radio_black)
        btn_slice = QPushButton("Apply Slicing")
        btn_slice.clicked.connect(self.w2_slicing)
        form.addWidget(btn_slice)

        # Bit Plane
        form.addWidget(QLabel("Bit Plane Slicing:"))
        self.w2_bits_checks = []
        bits_layout = QGridLayout()
        for i in range(8):
            chk = QCheckBox(f"Bit {i}")
            chk.setChecked(True) # Default all on
            chk.stateChanged.connect(self.w2_bit_plane)
            self.w2_bits_checks.append(chk)
            bits_layout.addWidget(chk, i//4, i%4)
        form.addLayout(bits_layout)

        form.addStretch()
        ctrl_panel.setLayout(form)

        # Display
        self.w2_display = QLabel()
        self.w2_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        scroll = QScrollArea()
        scroll.setWidget(self.w2_display)
        scroll.setWidgetResizable(True)

        layout.addWidget(ctrl_panel, 1)
        layout.addWidget(scroll, 3)

    def get_gray_w2(self):
        if self.original_image is None: return None
        return cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)

    def w2_update_display(self, img):
        self.w2_display.setPixmap(self.cv_to_qt_pixmap(img))

    def w2_invert(self):
        img = self.get_gray_w2()
        if img is None: return
        res = cv2.bitwise_not(img)
        self.w2_update_display(res)

    def w2_threshold(self):
        img = self.get_gray_w2()
        if img is None: return
        thresh = self.w2_slider_thresh.value()
        _, res = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)
        self.w2_update_display(res)

    def w2_log(self):
        img = self.get_gray_w2()
        if img is None: return
        # c * log(1 + r)
        c = 255 / np.log(1 + np.max(img))
        log_image = c * (np.log(img + 1))
        self.w2_update_display(np.array(log_image, dtype=np.uint8))

    def w2_gamma(self):
        img = self.get_gray_w2()
        if img is None: return
        gamma = self.w2_slider_gamma.value() / 10.0
        # c * r^gamma
        gamma_corrected = np.array(255 * (img / 255) ** gamma, dtype=np.uint8)
        self.w2_update_display(gamma_corrected)

    def w2_slicing(self):
        img = self.get_gray_w2()
        if img is None: return
        min_v = self.w2_spin_min.value()
        max_v = self.w2_spin_max.value()
        
        rows, cols = img.shape
        res = img.copy()
        
        if self.w2_radio_black.isChecked():
            res[:,:] = 0 # Black BG logic base
            mask = (img >= min_v) & (img <= max_v)
            res[mask] = 255
        else:
            # Keep BG, highlight range
            mask = (img >= min_v) & (img <= max_v)
            res[mask] = 255
            
        self.w2_update_display(res)

    def w2_bit_plane(self):
        img = self.get_gray_w2()
        if img is None: return
        
        res = np.zeros_like(img)
        for i in range(8):
            if self.w2_bits_checks[i].isChecked():
                # Add contribution of this bit
                layer = (img >> i) & 1
                res += layer * (2**i)
        
        self.w2_update_display(res)

    # ================= WEEK 3: Histogram =================
    def setup_week3(self):
        layout = QVBoxLayout(self.tab_w3)
        
        # Toolbar
        toolbar = QHBoxLayout()
        btn_hist = QPushButton("Show Histogram")
        btn_hist.clicked.connect(self.w3_show_hist)
        btn_eq = QPushButton("Equalize Histogram")
        btn_eq.clicked.connect(self.w3_equalize)
        
        btn_match_load = QPushButton("Load Reference (Match)")
        btn_match_load.clicked.connect(self.w3_load_ref)
        self.btn_match_apply = QPushButton("Apply Matching")
        self.btn_match_apply.setEnabled(False)
        self.btn_match_apply.clicked.connect(self.w3_match)
        
        toolbar.addWidget(btn_hist)
        toolbar.addWidget(btn_eq)
        toolbar.addWidget(btn_match_load)
        toolbar.addWidget(self.btn_match_apply)
        layout.addLayout(toolbar)

        # Content Area: Splitter or Grid
        # Left: Original, Right: Processed. Bottom: Histograms
        content = QGridLayout()
        
        self.w3_lbl_orig = QLabel("Original")
        self.w3_lbl_res = QLabel("Result")
        self.w3_lbl_orig.setScaledContents(True)
        self.w3_lbl_res.setScaledContents(True)
        self.w3_lbl_orig.setFixedSize(300, 300)
        self.w3_lbl_res.setFixedSize(300, 300)
        
        self.w3_hist_orig = HistogramCanvas(width=4, height=3)
        self.w3_hist_res = HistogramCanvas(width=4, height=3)

        content.addWidget(self.w3_lbl_orig, 0, 0)
        content.addWidget(self.w3_lbl_res, 0, 1)
        content.addWidget(self.w3_hist_orig, 1, 0)
        content.addWidget(self.w3_hist_res, 1, 1)
        
        layout.addLayout(content)
        self.w3_ref_img = None

    def w3_plot(self, img, canvas):
        canvas.axes.clear()
        if len(img.shape) == 2:
            canvas.axes.hist(img.ravel(), 256, [0, 256], color='gray')
        else:
            color = ('b', 'g', 'r')
            for i, col in enumerate(color):
                histr = cv2.calcHist([img], [i], None, [256], [0, 256])
                canvas.axes.plot(histr, color=col)
        canvas.draw()

    def w3_show_hist(self):
        if self.original_image is None: return
        self.w3_lbl_orig.setPixmap(self.cv_to_qt_pixmap(self.original_image))
        self.w3_plot(self.original_image, self.w3_hist_orig)
        # Clear result side
        self.w3_lbl_res.clear()
        self.w3_hist_res.axes.clear(); self.w3_hist_res.draw()

    def w3_equalize(self):
        if self.original_image is None: return
        self.w3_show_hist() # Update left side
        
        img_yuv = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2YUV)
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0]) # Equalize Y channel
        res = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        
        self.w3_lbl_res.setPixmap(self.cv_to_qt_pixmap(res))
        self.w3_plot(res, self.w3_hist_res)

    def w3_load_ref(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Load Reference Image")
        if fname:
            self.w3_ref_img = cv2.imread(fname)
            self.btn_match_apply.setEnabled(True)
            QMessageBox.information(self, "Info", "Reference image loaded!")

    def w3_match(self):
        if self.original_image is None or self.w3_ref_img is None: return
        
        # Using a simple implementation for Grayscale matching logic mapped to RGB
        # For simplicity, we process each channel independently
        source = self.original_image
        template = self.w3_ref_img
        
        # Resize template to match source for calculations might be heavy, 
        # but histogram matching works on distribution, not size.
        
        matched = np.zeros_like(source)
        for i in range(3):
            src_chan = source[:,:,i]
            tmpl_chan = template[:,:,i]
            
            # Compute CDFs
            hist_src, _ = np.histogram(src_chan.flatten(), 256, [0,256])
            cdf_src = hist_src.cumsum().astype(np.float64)
            cdf_src /= cdf_src[-1]
            
            hist_tmpl, _ = np.histogram(tmpl_chan.flatten(), 256, [0,256])
            cdf_tmpl = hist_tmpl.cumsum().astype(np.float64)
            cdf_tmpl /= cdf_tmpl[-1]
            
            # Mapping
            interp_vals = np.interp(cdf_src, cdf_tmpl, np.arange(256))
            matched[:,:,i] = interp_vals[src_chan].astype(np.uint8)
            
        self.w3_lbl_res.setPixmap(self.cv_to_qt_pixmap(matched))
        self.w3_plot(matched, self.w3_hist_res)
        self.w3_lbl_orig.setPixmap(self.cv_to_qt_pixmap(source))
        self.w3_plot(source, self.w3_hist_orig)


    # ================= WEEK 4: Spatial Filtering =================
    def setup_week4(self):
        layout = QHBoxLayout(self.tab_w4)
        
        # Left Control
        panel = QGroupBox("Spatial Filters")
        vbox = QVBoxLayout()
        
        # 1. Linear
        vbox.addWidget(QLabel("<b>1. Linear Smoothing</b>"))
        self.w4_combo_linear = QComboBox()
        self.w4_combo_linear.addItems(["Mean (Box)", "Gaussian"])
        vbox.addWidget(self.w4_combo_linear)
        
        vbox.addWidget(QLabel("Kernel Size (odd):"))
        self.w4_spin_ksize = QSpinBox(); self.w4_spin_ksize.setRange(3, 35); self.w4_spin_ksize.setSingleStep(2)
        vbox.addWidget(self.w4_spin_ksize)
        
        btn_linear = QPushButton("Apply Linear Filter")
        btn_linear.clicked.connect(self.w4_linear)
        vbox.addWidget(btn_linear)
        
        # 2. Non-Linear
        vbox.addWidget(QLabel("<b>2. Non-Linear (Noise)</b>"))
        btn_noise = QPushButton("Add Salt & Pepper (Demo)")
        btn_noise.clicked.connect(self.w4_add_noise)
        vbox.addWidget(btn_noise)
        
        self.w4_combo_nonlinear = QComboBox()
        self.w4_combo_nonlinear.addItems(["Median", "Max", "Min"])
        vbox.addWidget(self.w4_combo_nonlinear)
        
        btn_nonlinear = QPushButton("Apply Non-Linear")
        btn_nonlinear.clicked.connect(self.w4_nonlinear)
        vbox.addWidget(btn_nonlinear)
        
        # 3. Sharpening
        vbox.addWidget(QLabel("<b>3. Sharpening</b>"))
        self.w4_combo_sharp = QComboBox()
        self.w4_combo_sharp.addItems(["Laplacian Basic", "Laplacian Diagonal"])
        vbox.addWidget(self.w4_combo_sharp)
        btn_sharp = QPushButton("Apply Sharpening")
        btn_sharp.clicked.connect(self.w4_sharpen)
        vbox.addWidget(btn_sharp)
        
        vbox.addStretch()
        panel.setLayout(vbox)
        
        # Display
        self.w4_display = QLabel()
        self.w4_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        scroll = QScrollArea()
        scroll.setWidget(self.w4_display)
        scroll.setWidgetResizable(True)
        
        layout.addWidget(panel, 1)
        layout.addWidget(scroll, 3)

    def w4_linear(self):
        if self.original_image is None: return
        k = self.w4_spin_ksize.value()
        mode = self.w4_combo_linear.currentText()
        if mode == "Mean (Box)":
            res = cv2.blur(self.original_image, (k, k))
        else: # Gaussian
            res = cv2.GaussianBlur(self.original_image, (k, k), 0)
        self.w4_display.setPixmap(self.cv_to_qt_pixmap(res))
        
    def w4_add_noise(self):
        if self.original_image is None: return
        row, col, ch = self.original_image.shape
        s_vs_p = 0.5
        amount = 0.04
        out = np.copy(self.original_image)
        # Salt
        num_salt = np.ceil(amount * out.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in out.shape]
        out[coords[0], coords[1], :] = 255
        # Pepper
        num_pepper = np.ceil(amount * out.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in out.shape]
        out[coords[0], coords[1], :] = 0
        self.w4_display.setPixmap(self.cv_to_qt_pixmap(out))
        # Keep this noisy image for processing? Ideally yes, but for simplicity we filter original.
        # Let's temporarily update original so user can filter it? No, better use a temp variable.
        self.temp_noisy = out

    def w4_nonlinear(self):
        # Use temp noisy if exists, else original
        src = getattr(self, 'temp_noisy', self.original_image)
        if src is None: return
        
        k = self.w4_spin_ksize.value()
        mode = self.w4_combo_nonlinear.currentText()
        
        if mode == "Median":
            if k % 2 == 0: k += 1
            res = cv2.medianBlur(src, k)
        elif mode == "Max":
            kernel = np.ones((k,k), np.uint8)
            res = cv2.dilate(src, kernel) # Max filter is essentially dilation
        elif mode == "Min":
            kernel = np.ones((k,k), np.uint8)
            res = cv2.erode(src, kernel) # Min filter is essentially erosion
            
        self.w4_display.setPixmap(self.cv_to_qt_pixmap(res))

    def w4_sharpen(self):
        if self.original_image is None: return
        mode = self.w4_combo_sharp.currentText()
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        
        if "Diagonal" in mode:
            kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]]) # Standard diagonal laplacian
        else:
            kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
            
        # Filter
        res = cv2.filter2D(gray, cv2.CV_64F, kernel)
        # Convert back to uint8, scaling absolute values or adding back to original?
        # Usually Sharpened = Original - Laplacian (if center negative)
        # Or Original + Laplacian (if center positive)
        
        # Visualizing just the edges (Laplacian)
        abs_res = cv2.convertScaleAbs(res)
        self.w4_display.setPixmap(self.cv_to_qt_pixmap(abs_res))


    # ================= WEEK 5: Frequency Domain =================
    def setup_week5(self):
        layout = QHBoxLayout(self.tab_w5)
        
        controls = QGroupBox("Frequency Controls")
        vbox = QVBoxLayout()
        
        btn_fft = QPushButton("1. Convert to FFT & Show Spectrum")
        btn_fft.clicked.connect(self.w5_show_spectrum)
        vbox.addWidget(btn_fft)
        
        vbox.addWidget(QLabel("Filter Type:"))
        self.w5_combo_type = QComboBox(); self.w5_combo_type.addItems(["Lowpass", "Highpass"])
        vbox.addWidget(self.w5_combo_type)
        
        vbox.addWidget(QLabel("Model:"))
        self.w5_combo_model = QComboBox(); self.w5_combo_model.addItems(["Ideal", "Butterworth", "Gaussian"])
        vbox.addWidget(self.w5_combo_model)
        
        vbox.addWidget(QLabel("Cut-off (D0):"))
        self.w5_slider_d0 = QSlider(Qt.Orientation.Horizontal); self.w5_slider_d0.setRange(1, 200); self.w5_slider_d0.setValue(30)
        vbox.addWidget(self.w5_slider_d0)
        
        vbox.addWidget(QLabel("Order (n) - Butterworth:"))
        self.w5_spin_n = QSpinBox(); self.w5_spin_n.setValue(2)
        vbox.addWidget(self.w5_spin_n)
        
        self.w5_chk_dc = QCheckBox("Remove DC (F0,0 = 0)")
        vbox.addWidget(self.w5_chk_dc)
        
        btn_apply = QPushButton("2. Apply Filter (IFFT)")
        btn_apply.clicked.connect(self.w5_apply)
        vbox.addWidget(btn_apply)
        vbox.addStretch()
        controls.setLayout(vbox)
        
        # Display Area: Original, Spectrum, Result
        self.w5_lbl_spec = QLabel("Spectrum")
        self.w5_lbl_spec.setScaledContents(True); self.w5_lbl_spec.setFixedSize(300, 300)
        self.w5_lbl_res = QLabel("Result")
        self.w5_lbl_res.setScaledContents(True); self.w5_lbl_res.setFixedSize(300, 300)
        
        right_layout = QVBoxLayout()
        right_layout.addWidget(QLabel("Spectrum (Log Magnitude)"))
        right_layout.addWidget(self.w5_lbl_spec)
        right_layout.addWidget(QLabel("Result Spatial"))
        right_layout.addWidget(self.w5_lbl_res)
        
        layout.addWidget(controls, 1)
        layout.addLayout(right_layout, 2)

    def w5_show_spectrum(self):
        if self.original_image is None: return
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        f = np.fft.fft2(gray)
        self.fshift = np.fft.fftshift(f)
        
        # Magnitude for display
        magnitude = 20 * np.log(np.abs(self.fshift) + 1)
        magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        self.w5_lbl_spec.setPixmap(self.cv_to_qt_pixmap(magnitude))

    def w5_apply(self):
        if not hasattr(self, 'fshift') or self.fshift is None: 
            self.w5_show_spectrum()
            if not hasattr(self, 'fshift'): return

        rows, cols = self.fshift.shape
        crow, ccol = rows//2, cols//2
        
        # Create Mask
        mask = np.zeros((rows, cols, 2), np.float32) # For 2 channels if using OpenCV dft, but numpy fft uses complex
        # Numpy mask
        d0 = self.w5_slider_d0.value()
        n = self.w5_spin_n.value()
        ftype = self.w5_combo_type.currentText()
        model = self.w5_combo_model.currentText()
        
        u, v = np.meshgrid(np.arange(cols), np.arange(rows))
        u = u - ccol
        v = v - crow
        D = np.sqrt(u**2 + v**2)
        
        H = np.ones((rows, cols), dtype=np.float32)
        
        # Construct Filter H
        if model == "Ideal":
            if ftype == "Lowpass":
                H[D > d0] = 0
            else:
                H[D <= d0] = 0
        elif model == "Gaussian":
            if ftype == "Lowpass":
                H = np.exp(-(D**2) / (2 * d0**2))
            else:
                H = 1 - np.exp(-(D**2) / (2 * d0**2))
        elif model == "Butterworth":
            if ftype == "Lowpass":
                H = 1 / (1 + (D/d0)**(2*n))
            else:
                H = 1 / (1 + (d0/(D+1e-5))**(2*n))

        # Remove DC if checked
        if self.w5_chk_dc.isChecked():
            # In shifted FFT, DC is at crow, ccol
            # Simple notch at center
            H[crow-1:crow+2, ccol-1:ccol+2] = 0

        # Apply
        fshift_filtered = self.fshift * H
        
        # Inverse
        f_ishift = np.fft.ifftshift(fshift_filtered)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)
        
        img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        self.w5_lbl_res.setPixmap(self.cv_to_qt_pixmap(img_back))

    # ================= WEEK 6: PCA/Face =================
    # NOTE: Simplified version without external Dataset requirement logic 
    # User loads a folder of images.
    def setup_week6(self):
        layout = QVBoxLayout(self.tab_w6)
        
        h = QHBoxLayout()
        btn_train = QPushButton("1. Load Folder & Train PCA")
        btn_train.clicked.connect(self.w6_train)
        self.lbl_train_status = QLabel("Not Trained")
        h.addWidget(btn_train)
        h.addWidget(self.lbl_train_status)
        layout.addLayout(h)
        
        # Reconstruction
        layout.addWidget(QLabel("2. Reconstruction"))
        h2 = QHBoxLayout()
        self.w6_slider_k = QSlider(Qt.Orientation.Horizontal)
        self.w6_slider_k.setRange(1, 10)
        self.w6_slider_k.valueChanged.connect(self.w6_reconstruct)
        h2.addWidget(QLabel("K Components:"))
        h2.addWidget(self.w6_slider_k)
        layout.addLayout(h2)
        
        h3 = QHBoxLayout()
        self.w6_lbl_mean = QLabel("Mean Face"); self.w6_lbl_mean.setFixedSize(150, 150); self.w6_lbl_mean.setScaledContents(True)
        self.w6_lbl_recon = QLabel("Reconstructed"); self.w6_lbl_recon.setFixedSize(150, 150); self.w6_lbl_recon.setScaledContents(True)
        h3.addWidget(self.w6_lbl_mean)
        h3.addWidget(self.w6_lbl_recon)
        layout.addLayout(h3)
        
        # Recognition
        layout.addWidget(QLabel("3. Recognition"))
        btn_rec = QPushButton("Load Test Image & Recognize")
        btn_rec.clicked.connect(self.w6_recognize)
        layout.addWidget(btn_rec)
        self.w6_lbl_result = QLabel("Result: ...")
        layout.addWidget(self.w6_lbl_result)

    def w6_train(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Face Dataset Folder")
        if not folder: return
        
        # Load images
        self.w6_images = []
        self.w6_names = []
        target_size = (64, 64) # Resize for consistency
        
        try:
            for fname in os.listdir(folder):
                path = os.path.join(folder, fname)
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, target_size)
                    self.w6_images.append(img.flatten())
                    self.w6_names.append(fname)
            
            if not self.w6_images:
                self.lbl_train_status.setText("No images found.")
                return

            # PCA Calculation
            self.w6_img_shape = target_size
            X = np.array(self.w6_images) # (N, Features)
            self.mean_face = np.mean(X, axis=0)
            X_centered = X - self.mean_face
            
            # SVD
            U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
            self.eigenfaces = Vt
            self.weights = np.dot(X_centered, self.eigenfaces.T)
            
            # Display Mean Face
            mean_img = self.mean_face.reshape(self.w6_img_shape).astype(np.uint8)
            self.w6_lbl_mean.setPixmap(self.cv_to_qt_pixmap(mean_img))
            
            self.w6_slider_k.setRange(1, len(self.w6_images))
            self.lbl_train_status.setText(f"Trained on {len(self.w6_images)} images.")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def w6_reconstruct(self):
        if not hasattr(self, 'mean_face'): return
        k = self.w6_slider_k.value()
        
        # Reconstruct the LAST loaded training image as demo
        if not self.w6_images: return
        img_flat = self.w6_images[0]
        
        # Project
        diff = img_flat - self.mean_face
        w = np.dot(diff, self.eigenfaces[:k].T)
        
        # Reconstruct
        recon = self.mean_face + np.dot(w, self.eigenfaces[:k])
        recon_img = recon.reshape(self.w6_img_shape).astype(np.uint8)
        self.w6_lbl_recon.setPixmap(self.cv_to_qt_pixmap(recon_img))

    def w6_recognize(self):
        if not hasattr(self, 'weights'): 
            QMessageBox.warning(self, "Warning", "Train Model First")
            return
            
        fname, _ = QFileDialog.getOpenFileName(self, "Select Test Image")
        if not fname: return
        
        img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, self.w6_img_shape)
        img_flat = img.flatten()
        
        # Project
        diff = img_flat - self.mean_face
        w_test = np.dot(diff, self.eigenfaces.T) # Using all components for recognition
        
        # Find nearest
        min_dist = float('inf')
        best_match_idx = -1
        
        for i, w_train in enumerate(self.weights):
            dist = np.linalg.norm(w_test - w_train)
            if dist < min_dist:
                min_dist = dist
                best_match_idx = i
                
        # Threshold for Unknown (Arbitrary for demo)
        threshold = 3000000 # Depends on data scale, hard to guess without norm
        # Just show result
        name = self.w6_names[best_match_idx]
        self.w6_lbl_result.setText(f"Best Match: {name}\nDistance: {min_dist:.2f}")

    # ================= WEEK 7: Restoration =================
    def setup_week7(self):
        layout = QHBoxLayout(self.tab_w7)
        
        # Sidebar
        scroll_w = QScrollArea()
        panel = QGroupBox("Restoration Tools")
        vbox = QVBoxLayout()
        
        # Degradation
        vbox.addWidget(QLabel("<b>1. Degrade (Add Noise)</b>"))
        self.w7_combo_noise = QComboBox(); self.w7_combo_noise.addItems(["Gaussian", "Salt&Pepper", "Periodic"])
        vbox.addWidget(self.w7_combo_noise)
        btn_degrade = QPushButton("Apply Noise")
        btn_degrade.clicked.connect(self.w7_add_noise)
        vbox.addWidget(btn_degrade)
        
        # Filters (Spatial)
        vbox.addWidget(QLabel("<b>2. Spatial Denoise</b>"))
        self.w7_combo_filter = QComboBox()
        self.w7_combo_filter.addItems(["Arithmetic Mean", "Geometric Mean", "Median", "Harmonic Mean", "Contra-Harmonic"])
        vbox.addWidget(self.w7_combo_filter)
        
        vbox.addWidget(QLabel("Q (Contra-harmonic):"))
        self.w7_spin_q = QSpinBox(); self.w7_spin_q.setRange(-5, 5); self.w7_spin_q.setValue(1)
        vbox.addWidget(self.w7_spin_q)
        
        btn_denoise = QPushButton("Apply Denoise")
        btn_denoise.clicked.connect(self.w7_denoise)
        vbox.addWidget(btn_denoise)
        
        # Periodic / Inverse
        vbox.addWidget(QLabel("<b>3. Freq Restoration</b>"))
        btn_notch = QPushButton("Notch Filter (Auto)")
        btn_notch.clicked.connect(self.w7_notch)
        vbox.addWidget(btn_notch)
        
        vbox.addStretch()
        panel.setLayout(vbox)
        scroll_w.setWidget(panel); scroll_w.setWidgetResizable(True)
        
        # Display
        self.w7_display = QLabel(); self.w7_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        disp_scroll = QScrollArea(); disp_scroll.setWidget(self.w7_display); disp_scroll.setWidgetResizable(True)
        
        layout.addWidget(scroll_w, 1)
        layout.addWidget(disp_scroll, 3)
        
        self.w7_noisy_img = None

    def w7_add_noise(self):
        if self.original_image is None: return
        img = self.original_image.copy()
        mode = self.w7_combo_noise.currentText()
        
        if mode == "Gaussian":
            gauss = np.random.normal(0, 25, img.shape).astype('uint8')
            img = cv2.add(img, gauss)
        elif mode == "Salt&Pepper":
            # Reuse logic from week 4 but heavier
            prob = 0.05
            thres = 1 - prob
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    rdn = np.random.random()
                    if rdn < prob: img[i][j] = 0
                    elif rdn > thres: img[i][j] = 255
        elif mode == "Periodic":
            # Add Sinusoidal noise
            h, w = img.shape[:2]
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            x = np.arange(w); y = np.arange(h)
            X, Y = np.meshgrid(x, y)
            noise = 50 * np.sin(2 * np.pi * (X/20 + Y/20))
            img = cv2.add(img_gray, noise.astype(np.uint8))
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        self.w7_noisy_img = img
        self.w7_display.setPixmap(self.cv_to_qt_pixmap(img))

    def w7_denoise(self):
        if self.w7_noisy_img is None: return
        img = self.w7_noisy_img
        # Convert to float for calculation
        img_f = img.astype(np.float32)
        mode = self.w7_combo_filter.currentText()
        k = 3 # Kernel size
        
        res = img.copy()
        
        if mode == "Arithmetic Mean":
            res = cv2.blur(img, (k,k))
        elif mode == "Median":
            res = cv2.medianBlur(img, k)
        elif mode == "Geometric Mean":
            # Prod^(1/mn)
            # Log trick: exp(mean(log(img)))
            img_f[img_f==0] = 0.001
            log_img = np.log(img_f)
            blur_log = cv2.blur(log_img, (k,k))
            res = np.exp(blur_log).astype(np.uint8)
        elif mode == "Contra-Harmonic":
            # (Sum(f^(Q+1))) / (Sum(f^Q))
            Q = self.w7_spin_q.value()
            img_f[img_f==0] = 0.001 # Avoid div zero
            
            num = cv2.blur(np.power(img_f, Q+1), (k,k))
            den = cv2.blur(np.power(img_f, Q), (k,k))
            
            res_f = num / (den + 1e-5)
            res = np.clip(res_f, 0, 255).astype(np.uint8)

        self.w7_display.setPixmap(self.cv_to_qt_pixmap(res))

    def w7_notch(self):
        # Demo: Applies a fixed frequency domain mask assuming periodic noise added earlier
        if self.w7_noisy_img is None: return
        gray = cv2.cvtColor(self.w7_noisy_img, cv2.COLOR_BGR2GRAY)
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        
        # Mask center
        rows, cols = gray.shape
        crow, ccol = rows//2, cols//2
        
        # Masking specific spots where periodic noise usually appears (visual guess logic)
        # For this demo, we assume we know where the noise is (related to the X/20+Y/20 formula)
        # We just mask vertical/horizontal lines near center as a generic notch
        fshift[crow-5:crow+5, 0:ccol-10] = 0
        fshift[crow-5:crow+5, ccol+10:cols] = 0
        
        f_ishift = np.fft.ifftshift(fshift)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)
        img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        self.w7_display.setPixmap(self.cv_to_qt_pixmap(img_back))

    # ================= WEEK 8: Morphology =================
    def setup_week8(self):
        layout = QHBoxLayout(self.tab_w8)
        
        panel = QGroupBox("Morphology")
        vbox = QVBoxLayout()
        
        vbox.addWidget(QLabel("<b>1. Structuring Element</b>"))
        self.w8_combo_shape = QComboBox(); self.w8_combo_shape.addItems(["Rect", "Cross", "Ellipse"])
        vbox.addWidget(self.w8_combo_shape)
        vbox.addWidget(QLabel("Kernel Size:"))
        self.w8_spin_k = QSpinBox(); self.w8_spin_k.setRange(3, 21); self.w8_spin_k.setSingleStep(2)
        vbox.addWidget(self.w8_spin_k)
        
        vbox.addWidget(QLabel("<b>2. Basic Ops</b>"))
        btn_erode = QPushButton("Erosion"); btn_erode.clicked.connect(lambda: self.w8_op(cv2.MORPH_ERODE))
        vbox.addWidget(btn_erode)
        btn_dilate = QPushButton("Dilation"); btn_dilate.clicked.connect(lambda: self.w8_op(cv2.MORPH_DILATE))
        vbox.addWidget(btn_dilate)
        
        vbox.addWidget(QLabel("<b>3. Compound Ops</b>"))
        btn_open = QPushButton("Opening"); btn_open.clicked.connect(lambda: self.w8_op(cv2.MORPH_OPEN))
        vbox.addWidget(btn_open)
        btn_close = QPushButton("Closing"); btn_close.clicked.connect(lambda: self.w8_op(cv2.MORPH_CLOSE))
        vbox.addWidget(btn_close)
        
        vbox.addWidget(QLabel("<b>4. Applications</b>"))
        btn_bound = QPushButton("Boundary Extraction")
        btn_bound.clicked.connect(self.w8_boundary)
        vbox.addWidget(btn_bound)
        
        btn_count = QPushButton("Count Objects (Labeling)")
        btn_count.clicked.connect(self.w8_count)
        vbox.addWidget(btn_count)
        
        self.w8_lbl_info = QLabel("")
        vbox.addWidget(self.w8_lbl_info)
        
        vbox.addStretch()
        panel.setLayout(vbox)
        
        self.w8_display = QLabel(); self.w8_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        disp_scroll = QScrollArea(); disp_scroll.setWidget(self.w8_display); disp_scroll.setWidgetResizable(True)
        
        layout.addWidget(panel, 1)
        layout.addWidget(disp_scroll, 3)

    def get_binary_w8(self):
        if self.original_image is None: return None
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        return binary

    def w8_op(self, op_code):
        binary = self.get_binary_w8()
        if binary is None: return
        
        k = self.w8_spin_k.value()
        shape_txt = self.w8_combo_shape.currentText()
        if shape_txt == "Rect": shape = cv2.MORPH_RECT
        elif shape_txt == "Cross": shape = cv2.MORPH_CROSS
        else: shape = cv2.MORPH_ELLIPSE
        
        kernel = cv2.getStructuringElement(shape, (k, k))
        res = cv2.morphologyEx(binary, op_code, kernel)
        self.w8_display.setPixmap(self.cv_to_qt_pixmap(res))

    def w8_boundary(self):
        binary = self.get_binary_w8()
        if binary is None: return
        k = self.w8_spin_k.value()
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        erosion = cv2.erode(binary, kernel)
        boundary = binary - erosion
        self.w8_display.setPixmap(self.cv_to_qt_pixmap(boundary))

    def w8_count(self):
        binary = self.get_binary_w8()
        if binary is None: return
        
        # Use simple thresholding inversed if objects are black
        # Assuming objects are White on Black
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        
        # Map labels to colors
        output = np.zeros((binary.shape[0], binary.shape[1], 3), dtype=np.uint8)
        colors = np.random.randint(0, 255, size=(num_labels, 3), dtype=np.uint8)
        colors[0] = [0,0,0] # Background
        
        for i in range(1, num_labels):
            output[labels == i] = colors[i]
            
        self.w8_lbl_info.setText(f"Found: {num_labels - 1} objects")
        self.w8_display.setPixmap(self.cv_to_qt_pixmap(output))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CVApp()
    window.show()
    sys.exit(app.exec())