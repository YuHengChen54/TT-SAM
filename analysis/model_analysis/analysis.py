import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as metrics
import bisect
from matplotlib.patches import Rectangle


class Precision_Recall_Factory:
    """Calculate precision, recall, and F1 score metrics."""

    def calculate_precision_recall_f1(y_true, y_pred):
        from sklearn.metrics import precision_score, recall_score, f1_score
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        return precision, recall, f1

class TaiwanIntensity:
    """Taiwan seismic intensity scale with PGA and PGV thresholds."""
    label = ["0", "1", "2", "3", "4", "5-", "5+", "6-", "6+", "7"]
    pga = np.log10(
        [1e-5, 0.008, 0.025, 0.080, 0.250, 0.80, 1.4, 2.5, 4.4, 8.0]
    )  # log10(m/s^2)
    pgv = np.log10(
        [1e-5, 0.002, 0.007, 0.019, 0.057, 0.15, 0.3, 0.5, 0.8, 1.4]
    )  # log10(m/s)

    def __init__(self):
        self.pga_ticks = self.get_ticks(self.pga)
        self.pgv_ticks = self.get_ticks(self.pgv)

    def calculate(self, pga, pgv=None, label=False):
        pga_intensity = bisect.bisect(self.pga, pga) - 1
        intensity = pga_intensity

        if pga > self.pga[5] and pgv is not None:
            pgv_intensity = bisect.bisect(self.pgv, pgv) - 1
            if pgv_intensity > pga_intensity:
                intensity = pgv_intensity

        if label:
            return self.label[intensity]
        else:
            return intensity

    @staticmethod
    def get_ticks(array):
        ticks = np.cumsum(array, dtype=float)
        ticks[2:] = ticks[2:] - ticks[:-2]
        ticks = ticks[1:] / 2
        ticks = np.append(ticks, (ticks[-1] * 2 - ticks[-2]))
        return ticks

class Intensity_Plotter:
    """Plot true vs predicted intensity values with color-coded grid."""

    def plot_true_predicted(
        y_true,
        y_pred,
        aggregation_method="point",
        axes=None,
        axis_fontsize=20,
        point_size=2,
        target_metric="pgv",
        plot_title=None,
    ):
        """Create scatter plot comparing true and predicted intensity values."""

        if axes is None:
            fig = plt.figure(figsize=(10, 10), dpi=400)
            axes = fig.add_subplot(111)
        else:
            fig = axes.figure

        axes.set_aspect("equal")

        if aggregation_method == "mean":
            y_pred_point = np.sum(y_pred[:, :, 0] * y_pred[:, :, 1], axis=1)
        elif aggregation_method == "point":
            y_pred_point = y_pred
        else:
            raise ValueError(f'Aggregation type "{aggregation_method}" unknown')

        intensity = TaiwanIntensity()
        if target_metric == "pga":
            intensity_threshold = intensity.pga
            intensity_threshold[0] = np.log10(0.008)
            tick_positions = intensity.pga_ticks
        elif target_metric == "pgv":
            intensity_threshold = intensity.pgv
            intensity_threshold[0] = np.log10(0.002) - (np.log10(0.002) + 5) / 2
            tick_positions = intensity.pgv_ticks

        intensity_labels = intensity.label
        

        axis_limits = (intensity_threshold[0], intensity_threshold[-1])

        for i in range(len(intensity_threshold) - 1):
            for j in range(len(intensity_threshold) - 1):
                x_min, x_max = intensity_threshold[i], intensity_threshold[i + 1]
                y_min, y_max = intensity_threshold[j], intensity_threshold[j + 1]
                rectangle_color = "#eaffea"
                if abs(i - j) == 0:
                    rectangle_color = "#eaffea"
                elif abs(i - j) == 1:
                    rectangle_color = "#eaffea"
                elif j > i:
                    rectangle_color = "#ffecec"
                elif j < i:
                    rectangle_color = "#eef7ff"

                axes.add_patch(
                    Rectangle(
                        (x_min, y_min),
                        x_max - x_min,
                        y_max - y_min,
                        color=rectangle_color,
                        zorder=0
                    )
                )

        axes.scatter(
            y_true,
            y_pred_point,
            c="k",
            zorder=2,
            alpha=0.4,
            s=point_size
        )

        axes.plot(axis_limits, axis_limits, "k-", linewidth=1, zorder=3)

        for t in intensity_threshold:
            axes.plot([t, t], axis_limits, linestyle='dotted', color='grey', linewidth=1, zorder=4)
            axes.plot(axis_limits, [t, t], linestyle='dotted', color='grey', linewidth=1, zorder=4)

        label_y_position = axis_limits[1] 
        for i, label in enumerate(intensity_labels[:-1]):
            center_x = (intensity_threshold[i] + intensity_threshold[i + 1]) / 2
            axes.text(center_x, label_y_position, label, ha="center", va="bottom", fontsize=axis_fontsize-5, zorder=5)

        label_x_position = axis_limits[1] +0.02
        for i, label in enumerate(intensity_labels[:-1]):
            center_y = (intensity_threshold[i] + intensity_threshold[i + 1]) / 2
            axes.text(label_x_position, center_y, label, ha="left", va="center", fontsize=axis_fontsize-5, zorder=5)

        r2_score = metrics.r2_score(y_true, y_pred_point)
        axes.text(
            axis_limits[0] + 0.085,
            axis_limits[-1] - 0.08,
            f"$R^2={r2_score:.2f}$",
            va="top",
            fontsize=axis_fontsize - 4,
        )

        axes.set_xlim(axis_limits)
        axes.set_ylim(axis_limits)

        axes.set_xticks(intensity_threshold)
        axes.set_xticklabels([str(round(10**v, 3)) for v in intensity_threshold], fontsize=axis_fontsize - 7)
        axes.set_yticks(intensity_threshold)
        axes.set_yticklabels([str(round(10**v, 3)) for v in intensity_threshold], fontsize = axis_fontsize - 7)
        axes.set_clip_on(True)


        if plot_title is None:
            axes.set_title("Model prediction", fontsize=axis_fontsize + 5, pad=40)
        else:
            axes.set_title(plot_title, fontsize=axis_fontsize + 5, pad=40)

        if target_metric == "pga":
            axes.set_xlabel(r"True PGA (${m/s^2}$)", fontsize=axis_fontsize)
            axes.set_ylabel(r"Predicted PGA (${m/s^2}$)", fontsize=axis_fontsize)
        if target_metric == "pgv":
            axes.set_xlabel(r"True PGV (${m/s}$)", fontsize=axis_fontsize)
            axes.set_ylabel(r"Predicted PGV (${m/s}$)", fontsize=axis_fontsize)

        return fig, axes
