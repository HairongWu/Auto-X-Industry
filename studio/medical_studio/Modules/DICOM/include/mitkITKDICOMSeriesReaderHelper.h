/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#ifndef mitkITKDICOMSeriesReaderHelper_h
#define mitkITKDICOMSeriesReaderHelper_h

#include "mitkImage.h"
#include "mitkGantryTiltInformation.h"
#include "mitkDICOMTag.h"

#include <itkGDCMImageIO.h>

/* Forward deceleration of an DCMTK class. Used in the txx but part of the interface.*/
class OFDateTime;

namespace mitk
{

class ITKDICOMSeriesReaderHelper
{
  public:

    static const DICOMTag AcquisitionDateTag;
    static const DICOMTag AcquisitionTimeTag;
    static const DICOMTag TriggerTimeTag;

    typedef std::vector<std::string> StringContainer;
    typedef std::list<StringContainer> StringContainerList;

    Image::Pointer Load( const StringContainer& filenames, bool correctTilt, const GantryTiltInformation& tiltInfo );
    Image::Pointer Load3DnT( const StringContainerList& filenamesLists, bool correctTilt, const GantryTiltInformation& tiltInfo );

    static bool CanHandleFile(const std::string& filename);

  private:

    typedef std::vector<TimeBounds> TimeBoundsList;
    typedef itk::FixedArray<OFDateTime,2>  DateTimeBounds;


    /** Scans the given files for the acquisition time and returns the lowest and
     highest acquisition date time as date time bounds via bounds.
     @param bounds The acquisition date time bound extracted from the files.
     @param triggerBounds Time bounds for trigger information extracted from the files.
     If no trigger information was found than it returns trigger == [0.0, 0.0].
     @return If no acquisition date times can be found the function return will be false. Otherwise
     it returns True.
     */
    static bool ExtractDateTimeBoundsAndTriggerOfTimeStep( const StringContainer& filenamesOfTimeStep,
      DateTimeBounds& bounds, TimeBounds& triggerBounds);

    /* Determine the time bounds in ms respective to the baselineDateTime for the passed
    files. Additionally it regards the trigger time tag if set and acquisition date time
    carries not enough information.*/
    static bool ExtractTimeBoundsOfTimeStep(const StringContainer& filenamesOfTimeStep,
                                                 TimeBounds& bounds,
                                                 const OFDateTime& baselineDateTime );


    /** Returns the list of time bounds of all passed time step containers.
     (sa ExtractTimeBoundsOfTimeStep and ExtractDateTimeBoundsOfTimeStep).
     Time steps where no time bounds could be extracted
     are indicated by "null" time bounds (both values "0"). The order of the returned
     list equals of passed filenamesOfTimeSteps order.
     @remark The function regards acquisition date time tags and trigger time tags.*/
    static TimeBoundsList ExtractTimeBoundsOfTimeSteps (const StringContainerList& filenamesOfTimeSteps);

    /** Helper function that generates  a time geometry using the template and the passed boundslist
        (which indicates the number of time steps).
    */
    static TimeGeometry::Pointer GenerateTimeGeometry(const BaseGeometry* templateGeometry, const TimeBoundsList& boundsList);

    template <typename ImageType>
    typename ImageType::Pointer
    FixUpTiltedGeometry( ImageType* input, const GantryTiltInformation& tiltInfo );

    template <typename PixelType>
    Image::Pointer
    LoadDICOMByITK( const StringContainer& filenames,
                    bool correctTilt,
                    const GantryTiltInformation& tiltInfo,
                    itk::GDCMImageIO::Pointer& io);

    template <typename PixelType>
    Image::Pointer
    LoadDICOMByITK3DnT( const StringContainerList& filenames,
                        bool correctTilt,
                        const GantryTiltInformation& tiltInfo,
                        itk::GDCMImageIO::Pointer& io);


};

}

#endif
