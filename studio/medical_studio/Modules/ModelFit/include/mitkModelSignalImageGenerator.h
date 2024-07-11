/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#ifndef mitkModelSignalImageGenerator_h
#define mitkModelSignalImageGenerator_h


#include "mitkModelParameterizerBase.h"
#include "mitkImage.h"

#include "MitkModelFitExports.h"

namespace mitk
{
    /** Generator class that takes a model parameterizer instance, given parameter images and generates
     the corresponding signal image. Thus the generator simulates the signals of the model specified by
     parameterizer given the passed parameter images. The time grid of the signal is also defined by the
     parameterizer.*/
    class MITKMODELFIT_EXPORT ModelSignalImageGenerator: public ::itk::Object
    {
    public:

        mitkClassMacroItkParent(ModelSignalImageGenerator, itk::Object);
        itkFactorylessNewMacro(Self);

        typedef mitk::Image::Pointer ParameterImageType;
        typedef std::vector<std::string> ParameterNamesType;
        typedef unsigned int ParametersIndexType;
        typedef std::vector<ParameterImageType> ParameterVectorType;
        typedef std::map<ParametersIndexType, ParameterImageType> ParameterMapType;


        typedef mitk::Image::Pointer ResultImageType;
        typedef mitk::Image::Pointer MaskType;

        typedef mitk::ModelBase::TimeGridType GridType;

        itkSetObjectMacro(Parameterizer, ModelParameterizerBase);
        itkGetObjectMacro(Parameterizer, ModelParameterizerBase);

        void SetParameterInputImage(const ParametersIndexType index, ParameterImageType inputParameterImage);

        ResultImageType GetGeneratedImage();
        void Generate();

    protected:
        ModelSignalImageGenerator()
        {};
        ~ModelSignalImageGenerator() override
        {};

        template <typename TPixel, unsigned int VDim>
        void DoGenerateData(itk::Image<TPixel, VDim>* image);

        template <typename TPixel, unsigned int VDim>
        void DoPrepareMask(itk::Image<TPixel, VDim>* image);

    private:
        ParameterMapType m_ParameterInputMap;
        ParameterVectorType  m_InputParameterImages;

        void SortParameterImages();

        MaskType m_Mask;

        typedef itk::Image<unsigned char, 3> InternalMaskType;
        InternalMaskType::Pointer m_InternalMask;

        ResultImageType m_ResultImage;
        ModelParameterizerBase::Pointer m_Parameterizer;
    };
}


#endif
