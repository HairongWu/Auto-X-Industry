/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/
#ifndef QmitkHistogramVisualizationWidget_h
#define QmitkHistogramVisualizationWidget_h

#include <MitkImageStatisticsUIExports.h>
#include <ui_QmitkHistogramVisualizationWidget.h>

// itk
#include <itkHistogram.h>

/**
 * \brief Widget for displaying Histograms.
 */

class MITKIMAGESTATISTICSUI_EXPORT QmitkHistogramVisualizationWidget : public QWidget
{
  Q_OBJECT

public:
  QmitkHistogramVisualizationWidget(QWidget *parent = nullptr);
  /** \brief Draws the histogram and enables the GUI elements. */
  void SetHistogram(itk::Statistics::Histogram<double>::ConstPointer histogram, const std::string &dataLabel);
  /** \brief Clears the histogram and disables all GUI elements. */
  void Reset();
  /** \brief Sets the theme (either dark or light) */
  void SetTheme(QmitkChartWidget::ColorTheme style);
  /** \brief Resets the default settings */
  void ResetDefault();

  /** \brief Gets the number of bins to calculate */
  int GetBins();

 signals:
   /** \brief Signal to be emitted when the number of bins is changed by the user. The HistogramCalculator should
 connect to this signal and recalculate the data accordingly. */
  void RequestHistogramUpdate(unsigned int nBins);

private:
  void CreateConnections();
  void SetGUIElementsEnabled(bool enabled);
  /** \brief Helper function to convert the histogram in order to forward it to the ChartWidget. */
  std::vector<std::pair<double, double> > ConvertHistogramToPairList(itk::Statistics::Histogram<double>::ConstPointer histogram) const;

//slots
	/** \brief  Saves the histogram to the clipboard. */
	void OnClipboardButtonClicked();
	/** \brief Enables / Disables SpinBox to change the number of bins. */
	void OnDefaultNBinsCheckBoxChanged();
	/** \brief Emits the signal RequestHistogramUpdate(unsigned int nBins) with the updated value. */
	void OnNBinsSpinBoxValueChanged();
	/** \brief Shows / Hides the subchart. */
	void OnShowSubchartCheckBoxChanged();
	/** \brief Enables / Disables SpinBoxes to set custom min and max values */
	void OnViewMinMaxCheckBoxChanged();
	/**\brief */
	void OnMaxValueSpinBoxValueChanged();
	/** \brief */
	void OnMinValueSpinBoxValueChanged();

private:
  Ui::QmitkHistogramVisualizationControls m_Controls;
  const unsigned int m_DefaultNBins = 100;
  const unsigned int m_MinNBins = 10;
  const unsigned int m_MaxNBins = 10000;

  std::map < std::string, itk::Statistics::Histogram<double>::ConstPointer> m_Histograms;
};

#endif
