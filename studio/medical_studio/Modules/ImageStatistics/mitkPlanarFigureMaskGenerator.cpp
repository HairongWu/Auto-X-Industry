/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#include <mitkPlanarFigureMaskGenerator.h>
#include <mitkBaseGeometry.h>
#include <mitkITKImageImport.h>
#include "mitkImageAccessByItk.h"
#include <mitkExtractImageFilter.h>
#include <mitkConvert2Dto3DImageFilter.h>
#include <mitkImageTimeSelector.h>
#include <mitkIOUtil.h>

#include <itkCastImageFilter.h>
#include <itkVTKImageExport.h>
#include <itkVTKImageImport.h>
#include <itkImageDuplicator.h>
#include <itkMacro.h>
#include <itkLineIterator.h>

#include <vtkPoints.h>
#include <vtkImageStencil.h>
#include <vtkImageImport.h>
#include <vtkImageExport.h>
#include <vtkLassoStencilSource.h>
#include <vtkSmartPointer.h>



namespace mitk
{

void PlanarFigureMaskGenerator::SetPlanarFigure(mitk::PlanarFigure* planarFigure)
{
    if (nullptr == planarFigure )
    {
      throw std::runtime_error( "Error: planar figure empty!" );
    }

    const PlaneGeometry *planarFigurePlaneGeometry = planarFigure->GetPlaneGeometry();
    if ( planarFigurePlaneGeometry == nullptr )
    {
      throw std::runtime_error( "Planar-Figure not yet initialized!" );
    }

    const auto *planarFigureGeometry =
      dynamic_cast< const PlaneGeometry * >( planarFigurePlaneGeometry );
    if ( planarFigureGeometry == nullptr )
    {
      throw std::runtime_error( "Non-planar planar figures not supported!" );
    }

    if (planarFigure != m_PlanarFigure)
    {
        this->Modified();
        m_PlanarFigure = planarFigure;
    }

}

mitk::Image::ConstPointer PlanarFigureMaskGenerator::GetReferenceImage()
{
    if (IsUpdateRequired())
    {
        this->CalculateMask();
    }
    return m_ReferenceImage;
}

template < typename TPixel, unsigned int VImageDimension >
void PlanarFigureMaskGenerator::InternalCalculateMaskFromClosedPlanarFigure(
  const itk::Image< TPixel, VImageDimension > *image, unsigned int axis )
{
  typedef itk::Image< unsigned short, 2 > MaskImage2DType;

  typename MaskImage2DType::Pointer maskImage = MaskImage2DType::New();
  maskImage->SetOrigin(image->GetOrigin());
  maskImage->SetSpacing(image->GetSpacing());
  maskImage->SetLargestPossibleRegion(image->GetLargestPossibleRegion());
  maskImage->SetBufferedRegion(image->GetBufferedRegion());
  maskImage->SetDirection(image->GetDirection());
  maskImage->SetNumberOfComponentsPerPixel(image->GetNumberOfComponentsPerPixel());
  maskImage->Allocate();
  maskImage->FillBuffer(1);

  // all PolylinePoints of the PlanarFigure are stored in a vtkPoints object.
  // These points are used by the vtkLassoStencilSource to create
  // a vtkImageStencil.
  const mitk::PlaneGeometry *planarFigurePlaneGeometry = m_PlanarFigure->GetPlaneGeometry();
  const typename PlanarFigure::PolyLineType planarFigurePolyline = m_PlanarFigure->GetPolyLine( 0 );
  const mitk::BaseGeometry *imageGeometry3D = m_InputImage->GetGeometry( 0 );
  // If there is a second poly line in a closed planar figure, treat it as a hole.
  PlanarFigure::PolyLineType planarFigureHolePolyline;

  if (m_PlanarFigure->GetPolyLinesSize() == 2)
    planarFigureHolePolyline = m_PlanarFigure->GetPolyLine(1);


  // Determine x- and y-dimensions depending on principal axis
  // TODO use plane geometry normal to determine that automatically, then check whether the PF is aligned with one of the three principal axis
  int i0, i1;
  switch ( axis )
  {
  case 0:
    i0 = 1;
    i1 = 2;
    break;

  case 1:
    i0 = 0;
    i1 = 2;
    break;

  case 2:
  default:
    i0 = 0;
    i1 = 1;
    break;
  }

  // store the polyline contour as vtkPoints object
  vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
  for (const auto& point : planarFigurePolyline)
  {
    Point3D point3D;

    // Convert 2D point back to the local index coordinates of the selected image
    planarFigurePlaneGeometry->Map(point, point3D);
    imageGeometry3D->WorldToIndex(point3D, point3D);

    points->InsertNextPoint(point3D[i0], point3D[i1], 0);
  }

  vtkSmartPointer<vtkPoints> holePoints;

  if (!planarFigureHolePolyline.empty())
  {
    holePoints = vtkSmartPointer<vtkPoints>::New();
    Point3D point3D;

    for (const auto& point : planarFigureHolePolyline)
    {
      planarFigurePlaneGeometry->Map(point, point3D);
      imageGeometry3D->WorldToIndex(point3D, point3D);
      holePoints->InsertNextPoint(point3D[i0], point3D[i1], 0);
    }
  }

  // mark a malformed 2D planar figure ( i.e. area = 0 ) as out of bounds
  // this can happen when all control points of a rectangle lie on the same line = two of the three extents are zero
  double bounds[6] = {0};
  points->GetBounds(bounds);
  bool extent_x = (fabs(bounds[0] - bounds[1])) < mitk::eps;
  bool extent_y = (fabs(bounds[2] - bounds[3])) < mitk::eps;
  bool extent_z = (fabs(bounds[4] - bounds[5])) < mitk::eps;

  // throw an exception if a closed planar figure is deformed, i.e. has only one non-zero extent
  if (m_PlanarFigure->IsClosed() && ((extent_x && extent_y) || (extent_x && extent_z)  || (extent_y && extent_z)))
  {
    mitkThrow() << "Figure has a zero area and cannot be used for masking.";
  }

  // create a vtkLassoStencilSource and set the points of the Polygon
  vtkSmartPointer<vtkLassoStencilSource> lassoStencil = vtkSmartPointer<vtkLassoStencilSource>::New();
  lassoStencil->SetShapeToPolygon();
  lassoStencil->SetPoints(points);

  vtkSmartPointer<vtkLassoStencilSource> holeLassoStencil = nullptr;

  if (holePoints.GetPointer() != nullptr)
  {
    holeLassoStencil = vtkSmartPointer<vtkLassoStencilSource>::New();
    holeLassoStencil->SetShapeToPolygon();
    holeLassoStencil->SetPoints(holePoints);
  }

  // Export from ITK to VTK (to use a VTK filter)
  typedef itk::VTKImageImport< MaskImage2DType > ImageImportType;
  typedef itk::VTKImageExport< MaskImage2DType > ImageExportType;

  typename ImageExportType::Pointer itkExporter = ImageExportType::New();
  itkExporter->SetInput( maskImage );
//  itkExporter->SetInput( castFilter->GetOutput() );

  vtkSmartPointer<vtkImageImport> vtkImporter = vtkSmartPointer<vtkImageImport>::New();
  this->ConnectPipelines( itkExporter, vtkImporter );

  // Apply the generated image stencil to the input image
  vtkSmartPointer<vtkImageStencil> imageStencilFilter = vtkSmartPointer<vtkImageStencil>::New();
  imageStencilFilter->SetInputConnection( vtkImporter->GetOutputPort() );
  imageStencilFilter->SetStencilConnection(lassoStencil->GetOutputPort());
  imageStencilFilter->ReverseStencilOff();
  imageStencilFilter->SetBackgroundValue( 0 );
  imageStencilFilter->Update();

  vtkSmartPointer<vtkImageStencil> holeStencilFilter = nullptr;

  if (holeLassoStencil.GetPointer() != nullptr)
  {
    holeStencilFilter = vtkSmartPointer<vtkImageStencil>::New();
    holeStencilFilter->SetInputConnection(imageStencilFilter->GetOutputPort());
    holeStencilFilter->SetStencilConnection(holeLassoStencil->GetOutputPort());
    holeStencilFilter->ReverseStencilOn();
    holeStencilFilter->SetBackgroundValue(0);
    holeStencilFilter->Update();
  }

  // Export from VTK back to ITK
  vtkSmartPointer<vtkImageExport> vtkExporter = vtkSmartPointer<vtkImageExport>::New();
  vtkExporter->SetInputConnection( holeStencilFilter.GetPointer() == nullptr
    ? imageStencilFilter->GetOutputPort()
    : holeStencilFilter->GetOutputPort());
  vtkExporter->Update();

  typename ImageImportType::Pointer itkImporter = ImageImportType::New();
  this->ConnectPipelines( vtkExporter, itkImporter );
  itkImporter->Update();

  typedef itk::ImageDuplicator< ImageImportType::OutputImageType > DuplicatorType;
  DuplicatorType::Pointer duplicator = DuplicatorType::New();
  duplicator->SetInputImage( itkImporter->GetOutput() );
  duplicator->Update();

  // Store mask
  m_InternalITKImageMask2D = duplicator->GetOutput();
}

template < typename TPixel, unsigned int VImageDimension >
void PlanarFigureMaskGenerator::InternalCalculateMaskFromOpenPlanarFigure(
  const itk::Image< TPixel, VImageDimension > *image, unsigned int axis )
{
  typedef itk::Image< unsigned short, 2 >       MaskImage2DType;
  typedef itk::LineIterator< MaskImage2DType >  LineIteratorType;
  typedef MaskImage2DType::IndexType            IndexType2D;
  typedef std::vector< IndexType2D >            IndexVecType;

  typename MaskImage2DType::Pointer maskImage = MaskImage2DType::New();
  maskImage->SetOrigin(image->GetOrigin());
  maskImage->SetSpacing(image->GetSpacing());
  maskImage->SetLargestPossibleRegion(image->GetLargestPossibleRegion());
  maskImage->SetBufferedRegion(image->GetBufferedRegion());
  maskImage->SetDirection(image->GetDirection());
  maskImage->SetNumberOfComponentsPerPixel(image->GetNumberOfComponentsPerPixel());
  maskImage->Allocate();
  maskImage->FillBuffer(0);

  // all PolylinePoints of the PlanarFigure are stored in a vtkPoints object.
  const mitk::PlaneGeometry *planarFigurePlaneGeometry = m_PlanarFigure->GetPlaneGeometry();
  const typename PlanarFigure::PolyLineType planarFigurePolyline = m_PlanarFigure->GetPolyLine( 0 );
  const mitk::BaseGeometry *imageGeometry3D = m_InputImage->GetGeometry( 0 );

  // Determine x- and y-dimensions depending on principal axis
  // TODO use plane geometry normal to determine that automatically, then check whether the PF is aligned with one of the three principal axis
  int i0, i1;
  switch ( axis )
  {
  case 0:
    i0 = 1;
    i1 = 2;
    break;

  case 1:
    i0 = 0;
    i1 = 2;
    break;

  case 2:
  default:
    i0 = 0;
    i1 = 1;
    break;
  }

  int numPolyLines = m_PlanarFigure->GetPolyLinesSize();
  for ( int lineId = 0; lineId < numPolyLines; ++lineId )
  {
    // store the polyline contour as vtkPoints object
    IndexVecType pointIndices;
    for(const auto& point : planarFigurePolyline)
    {
      Point3D point3D;

      planarFigurePlaneGeometry->Map(point, point3D);
      imageGeometry3D->WorldToIndex(point3D, point3D);

      IndexType2D index2D;
      index2D[0] = point3D[i0];
      index2D[1] = point3D[i1];

      pointIndices.push_back( index2D );
    }

    size_t numLineSegments = pointIndices.size() - 1;
    for (size_t i = 0; i < numLineSegments; ++i)
    {
      LineIteratorType lineIt(maskImage, pointIndices[i], pointIndices[i+1]);
      while (!lineIt.IsAtEnd())
      {
        lineIt.Set(1);
        ++lineIt;
      }
    }
  }

  // Store mask
  m_InternalITKImageMask2D = maskImage;
}

bool PlanarFigureMaskGenerator::CheckPlanarFigureIsNotTilted(const PlaneGeometry* planarGeometry, const BaseGeometry *geometry)
{
  if (!planarGeometry) return false;
  if (!geometry) return false;

  unsigned int axis;
  return GetPrincipalAxis(geometry,planarGeometry->GetNormal(), axis);
}

bool PlanarFigureMaskGenerator::GetPrincipalAxis(
  const BaseGeometry *geometry, Vector3D vector,
  unsigned int &axis )
{
  vector.Normalize();
  for ( unsigned int i = 0; i < 3; ++i )
  {
    Vector3D axisVector = geometry->GetAxisVector( i );
    axisVector.Normalize();

    //normal mitk::eps is to pedantic for this check. See e.g. T27122
    //therefore choose a larger epsilon. The value was set a) as small as
    //possible but b) still allowing to datasets like in (T27122) to pass
    //when floating rounding errors sum up.
    const double epsilon = 5e-5;
    if ( fabs( fabs( axisVector * vector ) - 1.0) < epsilon)
    {
      axis = i;
      return true;
    }
  }

  return false;
}

void PlanarFigureMaskGenerator::CalculateMask()
{
    if (m_InputImage.IsNull())
    {
      mitkThrow() << "Image is not set.";
    }

    if (m_PlanarFigure.IsNull())
    {
      mitkThrow() << "PlanarFigure is not set.";
    }

    const BaseGeometry *imageGeometry = m_InputImage->GetGeometry();
    if ( imageGeometry == nullptr )
    {
      mitkThrow() << "Image geometry invalid!";
    }

    auto timePointImage = SelectImageByTimePoint(m_InputImage, m_TimePoint);

    if (timePointImage.IsNull()) mitkThrow() << "Cannot generate mask. Passed time point is not supported by input image.";

    m_InternalITKImageMask2D = nullptr;
    const PlaneGeometry *planarFigurePlaneGeometry = m_PlanarFigure->GetPlaneGeometry();
    const auto *planarFigureGeometry = dynamic_cast< const PlaneGeometry * >( planarFigurePlaneGeometry );

    // Find principal direction of PlanarFigure in input image
    unsigned int axis;
    if ( !this->GetPrincipalAxis( imageGeometry,
      planarFigureGeometry->GetNormal(), axis ) )
    {
      throw std::runtime_error( "Non-aligned planar figures not supported!" );
    }
    m_PlanarFigureAxis = axis;

    // Find slice number corresponding to PlanarFigure in input image
    itk::Image< unsigned short, 3 >::IndexType index;
    imageGeometry->WorldToIndex( planarFigureGeometry->GetOrigin(), index );

    unsigned int slice = index[axis];
    m_PlanarFigureSlice = slice;

    // extract image slice which corresponds to the planarFigure and store it in m_InternalImageSlice
    mitk::Image::ConstPointer inputImageSlice = Extract2DImageSlice(timePointImage, axis, slice);
    // Compute mask from PlanarFigure
    // rastering for open planar figure:
    if ( !m_PlanarFigure->IsClosed() )
    {
      AccessFixedDimensionByItk_1(inputImageSlice,
        InternalCalculateMaskFromOpenPlanarFigure,
        2, axis)
    }
    else//for closed planar figure
    {
      AccessFixedDimensionByItk_1(inputImageSlice,
                                  InternalCalculateMaskFromClosedPlanarFigure,
                                  2, axis)
    }

    //convert itk mask to mitk::Image::Pointer and return it
    mitk::Image::Pointer planarFigureMaskImage;
    planarFigureMaskImage = mitk::GrabItkImageMemory(m_InternalITKImageMask2D);

    m_ReferenceImage = inputImageSlice;
    m_InternalMask = planarFigureMaskImage;
}

unsigned int PlanarFigureMaskGenerator::GetNumberOfMasks() const
{
  return 1;
}

mitk::Image::ConstPointer PlanarFigureMaskGenerator::DoGetMask(unsigned int)
{
    if (IsUpdateRequired())
    {
        this->CalculateMask();
        this->Modified();
    }

    m_InternalMaskUpdateTime = this->GetMTime();
    return m_InternalMask;
}

mitk::Image::ConstPointer PlanarFigureMaskGenerator::Extract2DImageSlice(const Image* input, unsigned int axis, unsigned int slice) const
{
    // Extract slice with given position and direction from image
    unsigned int dimension = input->GetDimension();

    if (dimension == 3)
    {
      ExtractImageFilter::Pointer imageExtractor = ExtractImageFilter::New();
      imageExtractor->SetInput( input );
      imageExtractor->SetSliceDimension( axis );
      imageExtractor->SetSliceIndex( slice );
      imageExtractor->Update();
      return imageExtractor->GetOutput();
    }
    else if(dimension == 2)
    {
      return input;
    }
    else
    {
      MITK_ERROR << "Unsupported image dimension. Dimension is: " << dimension << ". Only 2D and 3D images are supported.";
      return nullptr;
    }
}

bool PlanarFigureMaskGenerator::IsUpdateRequired() const
{
    unsigned long thisClassTimeStamp = this->GetMTime();
    unsigned long internalMaskTimeStamp = m_InternalMask->GetMTime();
    unsigned long planarFigureTimeStamp = m_PlanarFigure->GetMTime();
    unsigned long inputImageTimeStamp = m_InputImage->GetMTime();

    if (thisClassTimeStamp > m_InternalMaskUpdateTime) // inputs have changed
    {
        return true;
    }

    if (m_InternalMaskUpdateTime < planarFigureTimeStamp || m_InternalMaskUpdateTime < inputImageTimeStamp) // mask image has changed outside of this class
    {
        return true;
    }

    if (internalMaskTimeStamp > m_InternalMaskUpdateTime) // internal mask has been changed outside of this class
    {
        return true;
    }

    return false;
}

}

