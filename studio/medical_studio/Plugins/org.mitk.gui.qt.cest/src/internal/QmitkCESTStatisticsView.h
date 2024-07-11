/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/


#ifndef QmitkCESTStatisticsView_h
#define QmitkCESTStatisticsView_h

#include <berryISelectionListener.h>

#include <QmitkAbstractView.h>
#include <QmitkSliceNavigationListener.h>

#include "ui_QmitkCESTStatisticsViewControls.h"
#include <QmitkImageStatisticsCalculationRunnable.h>

#include <mitkPlanarFigure.h>
#include <mitkPointSet.h>

#include <mitkIRenderWindowPartListener.h>

/**
  \brief QmitkCESTStatisticsView

  \warning  Basic statistics view for CEST data.

  \sa QmitkAbstractView
  \ingroup ${plugin_target}_internal
*/
class QmitkCESTStatisticsView : public QmitkAbstractView, public mitk::IRenderWindowPartListener
{
  Q_OBJECT

  public:

    static const std::string VIEW_ID;

    /*!
    \brief default constructor */
    QmitkCESTStatisticsView(QObject *parent = nullptr, const char *name = nullptr);
    /*!
    \brief default destructor */
    ~QmitkCESTStatisticsView() override;

  protected slots:

    /// \brief Called when the user clicks the GUI button
    void OnThreeDimToFourDimPushButtonClicked();

    /// \brief takes care of processing the computed data
    void OnThreadedStatisticsCalculationEnds();

    /// \brief Toggle whether or not the plot uses a fixed x range
    void OnFixedRangeCheckBoxToggled(bool state);

    /// \brief Adapt axis scale when manual ranges are set
    void OnFixedRangeDoubleSpinBoxChanged();

    /// \brief What to do if the crosshair moves
    void OnSliceChanged();

  protected:

    void CreateQtPartControl(QWidget *parent) override;

    void SetFocus() override;

    void RenderWindowPartActivated(mitk::IRenderWindowPart* renderWindowPart) override;
    void RenderWindowPartDeactivated(mitk::IRenderWindowPart* renderWindowPart) override;

    void OnSelectionChanged( berry::IWorkbenchPart::Pointer source,
                                     const QList<mitk::DataNode::Pointer>& nodes ) override;

    /// parse string and set data vector returns true if successful
    bool SetZSpectrum(mitk::StringProperty* zSpectrumProperty);

    /** Checks whether the currently set data appears reasonable
    */
    bool DataSanityCheck();

    /** Fills the plot based on a point set
    *
    * This will only use the first timestep
    */
    template <typename TPixel, unsigned int VImageDimension>
    void PlotPointSet(itk::Image<TPixel, VImageDimension>* image);

    /** Deletes all data
    */
    void Clear();

    /** Remove MZeros
    *
    * Will remove the data for the M0 images from the given input
    */
    void RemoveMZeros(QmitkPlotWidget::DataVector& xValues, QmitkPlotWidget::DataVector& yValues);
    void RemoveMZeros(QmitkPlotWidget::DataVector& xValues, QmitkPlotWidget::DataVector& yValues, QmitkPlotWidget::DataVector& stdDevs);

    /** Copies the first timestep of a segmentation to all others
    */
    template <typename TPixel, unsigned int VImageDimension>
    void CopyTimesteps(itk::Image<TPixel, VImageDimension>* image);

    Ui::QmitkCESTStatisticsViewControls m_Controls;
    QmitkImageStatisticsCalculationRunnable* m_CalculatorJob;
    QmitkPlotWidget::DataVector m_zSpectrum;
    mitk::Image::Pointer m_ZImage;
    mitk::Image::Pointer m_MaskImage;
    mitk::PlanarFigure::Pointer m_MaskPlanarFigure;
    mitk::PointSet::Pointer m_PointSet;
    mitk::PointSet::Pointer m_CrosshairPointSet;

    QmitkSliceNavigationListener m_SliceChangeListener;

    itk::TimeStamp m_selectedNodeTime;
    itk::TimeStamp m_currentPositionTime;
    /** @brief currently valid selected position in the inspector*/
    mitk::Point3D m_currentSelectedPosition;
    mitk::TimePointType m_currentSelectedTimePoint;

};

#endif
