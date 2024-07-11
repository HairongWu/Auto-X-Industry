/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#ifndef mitkExceptionMacro_h
#define mitkExceptionMacro_h

#include "mitkException.h"
#include <itkMacro.h>
#include <mitkLog.h>
#include <sstream>

/** The exception macro is used to throw an exception
 *  (i.e., usually a condition that results in program failure).
 *
 *  Example usage looks like:
 *  mitkThrow() << "this is error info";
 */
#define mitkThrow() throw mitk::Exception(__FILE__, __LINE__, "", ITK_LOCATION)

/** The rethrow macro is used to rethrow an existing exception. The
  * rethrow information (file,line of code) is then additionally stored
  * in the exception. To check if an exception was rethrown you can use
  * the methods GetNumberOfRethrows() and GetRethrowData().
  *
  * Example usage:
  * try
  *   {
  *   //some code that throws an exception
  *   }
  * catch(mitk::Exception e)
  *   {
  *   //here we want to rethrow the exception
  *   mitkReThrow(e) << "Message that will be appended to the exception (optional)";
  *   }
  */
#define mitkReThrow(mitkexception)                                                                                     \
  mitkexception.AddRethrowData(__FILE__, __LINE__, "Rethrow by mitkReThrow macro.");                                   \
  throw mitkexception

/** The specialized exception macro is used to throw exceptions
  * in cases of specialized errors. This means the second parameter must be a class which
  * inherits from mitk::Exception. An object of this exception is thrown when using the macro.
  * Thus, more differentiated excaptions can be thrown, when needed.
  *
  * Example usage:
  * mitkSpecializedExceptionMacro(mitk::MySpecializedException) << "this is error info";
  */
#define mitkThrowException(classname) throw classname(__FILE__, __LINE__, "", ITK_LOCATION)

/** Class macro for MITK exception classes.
  * All MITK exception classes should derive from MITK::Exception.
  */
#define mitkExceptionClassMacro(ClassName, SuperClassName)                                                             \
  ClassName(const char *file, unsigned int lineNumber, const char *desc, const char *loc)                              \
    : SuperClassName(file, lineNumber, desc, loc)                                                                      \
  {                                                                                                                    \
  }                                                                                                                    \
  itkTypeMacro(ClassName, SuperClassName);                                                                             \
  /** \brief Definition of the bit shift operator for this class. It can be used to add messages.*/                    \
  template <class T>                                                                                                   \
  inline ClassName &operator<<(const T &data)                                                                          \
  {                                                                                                                    \
    std::stringstream ss;                                                                                              \
    ss << this->GetDescription() << data;                                                                              \
    this->SetDescription(ss.str());                                                                                    \
    return *this;                                                                                                      \
  }                                                                                                                    \
  /** \brief Definition of the bit shift operator for this class (for non const data).*/                               \
  template <class T>                                                                                                   \
  inline ClassName &operator<<(T &data)                                                                                \
  {                                                                                                                    \
    std::stringstream ss;                                                                                              \
    ss << this->GetDescription() << data;                                                                              \
    this->SetDescription(ss.str());                                                                                    \
    return *this;                                                                                                      \
  }                                                                                                                    \
  /** \brief Definition of the bit shift operator for this class (for functions).*/                                    \
  inline ClassName &operator<<(std::ostream &(*func)(std::ostream &))                                                  \
  {                                                                                                                    \
    std::stringstream ss;                                                                                              \
    ss << this->GetDescription() << func;                                                                              \
    this->SetDescription(ss.str());                                                                                    \
    return *this;                                                                                                      \
  }

#endif
