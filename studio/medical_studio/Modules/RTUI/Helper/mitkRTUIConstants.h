/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#ifndef mitkRTUIConstants_h
#define mitkRTUIConstants_h

#include <string>
#include <mitkDoseValueType.h>

#include "MitkRTUIExports.h"

namespace mitk
{
struct MITKRTUI_EXPORT RTUIConstants
{
  /** ID/Path of main preference node for RT UI.  */
  static const std::string ROOT_PREFERENCE_NODE_ID;
  /** Bool that indicates how the prescribed dose should be defined, if unknown. True: UNKNOWN_PRESCRIBED_DOSE_HANDLING_VALUE should be used as
      default dose value in Gy; False: it should be used as fraction of the max dose to determine the prescribed dose.*/
  static const std::string UNKNOWN_PRESCRIBED_DOSE_HANDLING_AS_DEFAULT_ID;
  /** Value that is used to determine unknown prescribed doses.*/
  static const std::string UNKNOWN_PRESCRIBED_DOSE_HANDLING_VALUE_ID;

  /** ID/Path of main preference node where all iso dose level presets are stored (e.g. ROOT_ISO_PRESETS_PREFERENCE_NODE_ID+"/[Preset1]").  */
  static const std::string ROOT_ISO_PRESETS_PREFERENCE_NODE_ID;
  /** ID/Path of main preference for dose visualization preferences.  */
  static const std::string ROOT_DOSE_VIS_PREFERENCE_NODE_ID;
  /** ID for the reference dose stored as preference.  */
  static const std::string REFERENCE_DOSE_ID;
  /** ID for the preference flag that indicates if the reference dose is synced for all nodes*/
  static const std::string GLOBAL_REFERENCE_DOSE_SYNC_ID;
  /** ID for the flag if dose should be displayed as absolute dose.  */
  static const std::string DOSE_DISPLAY_ABSOLUTE_ID;
  /** ID for the global visibility switch for iso line visualization.  */
  static const std::string GLOBAL_VISIBILITY_ISOLINES_ID;
  /** ID for the global visibility switch for color wash visualization.  */
  static const std::string GLOBAL_VISIBILITY_COLORWASH_ID;
  /** ID for the selected iso preset that should be used (value of ROOT_ISO_PRESETS_PREFERENCE_NODE_ID + value of this key can
      be used to construct the passed to the selected preset.  */
  static const std::string SELECTED_ISO_PRESET_ID;

  /** ID for the relative dose value of an iso dose level.  */
  static const std::string ISO_LEVEL_DOSE_VALUE_ID;
  /** ID for the color (red component) of an iso dose level.  */
  static const std::string ISO_LEVEL_COLOR_RED_ID;
  /** ID for the color (green component) of an iso dose level.  */
  static const std::string ISO_LEVEL_COLOR_GREEN_ID;
  /** ID for the color (blue component) of an iso dose level.  */
  static const std::string ISO_LEVEL_COLOR_BLUE_ID;
  /** ID for the visibility switch for iso line visualization.  */
  static const std::string ISO_LEVEL_VISIBILITY_ISOLINES_ID;
  /** ID for the visibility switch for color wash visualization.  */
  static const std::string ISO_LEVEL_VISIBILITY_COLORWASH_ID;

  /** Default value used as reference_dose_if not defined by application or data node*/
  static const DoseValueAbs DEFAULT_REFERENCE_DOSE_VALUE;
};

struct MITKRTUI_EXPORT RTCTKEventConstants
{
  /** ID/Path of main preference node for RT UI.  */
  static const std::string TOPIC_REFERENCE_DOSE;
  static const std::string TOPIC_REFERENCE_DOSE_CHANGED;
  static const std::string TOPIC_ISO_DOSE_LEVEL_PRESETS;
  static const std::string TOPIC_ISO_DOSE_LEVEL_PRESETS_CHANGED;
  static const std::string TOPIC_GLOBAL_VISIBILITY_CHANGED;
};

}

#endif
