/* This file was generated by upbc (the upb compiler) from the input
 * file:
 *
 *     envoy/api/v2/listener/udp_listener_config.proto
 *
 * Do not edit -- your changes will be discarded when the file is
 * regenerated. */

#include <stddef.h>
#include "upb/msg.h"
#include "envoy/api/v2/listener/udp_listener_config.upb.h"
#include "google/protobuf/any.upb.h"
#include "google/protobuf/struct.upb.h"
#include "udpa/annotations/migrate.upb.h"

#include "upb/port_def.inc"

static const upb_msglayout *const envoy_api_v2_listener_UdpListenerConfig_submsgs[2] = {
  &google_protobuf_Any_msginit,
  &google_protobuf_Struct_msginit,
};

static const upb_msglayout_field envoy_api_v2_listener_UdpListenerConfig__fields[3] = {
  {1, UPB_SIZE(0, 0), 0, 0, 9, 1},
  {2, UPB_SIZE(8, 16), UPB_SIZE(-13, -25), 1, 11, 1},
  {3, UPB_SIZE(8, 16), UPB_SIZE(-13, -25), 0, 11, 1},
};

const upb_msglayout envoy_api_v2_listener_UdpListenerConfig_msginit = {
  &envoy_api_v2_listener_UdpListenerConfig_submsgs[0],
  &envoy_api_v2_listener_UdpListenerConfig__fields[0],
  UPB_SIZE(16, 32), 3, false,
};

const upb_msglayout envoy_api_v2_listener_ActiveRawUdpListenerConfig_msginit = {
  NULL,
  NULL,
  UPB_SIZE(0, 0), 0, false,
};

#include "upb/port_undef.inc"

