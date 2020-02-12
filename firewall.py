from pox.core import core
import pox.openflow.libopenflow_01 as of
from pox.lib.revent import *
from pox.lib.util import dpidToStr
from pox.lib.addresses import EthAddr
from collections import namedtuple
import os

log = core.getLogger()

class Firewall (EventMixin):

    def __init__ (self):
        self.listenTo(core.openflow)
        log.debug("Enabling Firewall Module")

    def _handle_ConnectionUp (self, event):
        ''' Add your logic here ... '''
        block = of.ofp_match()
        block.dl_src = EthAddr('00:00:00:00:00:02')
        block.dl_dst = EthAddr('00:00:00:00:00:03')
        flow_mod = of.ofp_flow_mod()
        flow_mod.match = block
        event.connection.send(flow_mod)
        log.debug("Firewall rules installed on %s", dpidToStr(event.dpid))

def launch ():
    core.registerNew(Firewall)