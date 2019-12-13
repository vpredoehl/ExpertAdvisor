drop table if exists audcad;
drop table if exists audchf;
drop table if exists audjpy;
drop table if exists audnzd;
drop table if exists audusd;
drop table if exists cadchf;
drop table if exists cadjpy;
drop table if exists chfjpy;
drop table if exists euraud;
drop table if exists eurcad;
drop table if exists eurchf;
drop table if exists eurgbp;
drop table if exists eurjpy;
drop table if exists eurnzd;
drop table if exists eurusd;
drop table if exists gbpaud;
drop table if exists gbpcad;
drop table if exists gbpchf;
drop table if exists gbpjpy;
drop table if exists gbpnzd;
drop table if exists gbpusd;
drop table if exists nzdcad;
drop table if exists nzdchf;
drop table if exists nzdjpy;
drop table if exists nzdusd;
drop table if exists usdcad;
drop table if exists usdchf;
drop table if exists usdjpy;

drop index if exists audcad_idx;
drop index if exists audchf_idx;
drop index if exists audjpy_idx;
drop index if exists audnzd_idx;
drop index if exists audusd_idx;
drop index if exists cadchf_idx;
drop index if exists cadjpy_idx;
drop index if exists chfjpy_idx;
drop index if exists euraud_idx;
drop index if exists eurcad_idx;
drop index if exists eurchf_idx;
drop index if exists eurgbp_idx;
drop index if exists eurjpy_idx;
drop index if exists eurnzd_idx;
drop index if exists eurusd_idx;
drop index if exists gbpaud_idx;
drop index if exists gbpcad_idx;
drop index if exists gbpchf_idx;
drop index if exists gbpjpy_idx;
drop index if exists gbpnzd_idx;
drop index if exists gbpusd_idx;
drop index if exists nzdcad_idx;
drop index if exists nzdchf_idx;
drop index if exists nzdjpy_idx;
drop index if exists nzdusd_idx;
drop index if exists usdcad_idx;
drop index if exists usdchf_idx;
drop index if exists usdjpy_idx;

create table audcad ( like audcadrmp ) partition by range (time);
create table audcad_2018 partition of audcad for values from ('2018-01-01') to ('2019-01-01');
create table audcad_2017 partition of audcad for values from ('2017-01-01') to ('2018-01-01')  tablespace fxarch;
create table audcad_2016 partition of audcad for values from ('2016-01-01') to ('2017-01-01')  tablespace fxarch;
create table audcad_2015 partition of audcad for values from ('2015-01-01') to ('2016-01-01')  tablespace fxarch;
create table audcad_2014 partition of audcad for values from ('2014-01-01') to ('2015-01-01')  tablespace fxarch;
create table audcad_2013 partition of audcad for values from ('2013-01-01') to ('2014-01-01')  tablespace fxarch;
create table audcad_2012 partition of audcad for values from ('2012-01-01') to ('2013-01-01')  tablespace fxarch;
create table audcad_2011 partition of audcad for values from ('2011-01-01') to ('2012-01-01')  tablespace fxarch;
create table audcad_2010 partition of audcad for values from ('2010-01-01') to ('2011-01-01')  tablespace fxarch;
create table audcadarch partition of audcad for values from ('2000-01-01') to ('2010-01-01')  tablespace fxarch;


create table audchf ( like audchfrmp ) partition by range (time);
create table audchf_2018 partition of audchf for values from ('2018-01-01') to ('2019-01-01');
create table audchf_2017 partition of audchf for values from ('2017-01-01') to ('2018-01-01')  tablespace fxarch;
create table audchf_2016 partition of audchf for values from ('2016-01-01') to ('2017-01-01')  tablespace fxarch;
create table audchf_2015 partition of audchf for values from ('2015-01-01') to ('2016-01-01')  tablespace fxarch;
create table audchf_2014 partition of audchf for values from ('2014-01-01') to ('2015-01-01')  tablespace fxarch;
create table audchf_2013 partition of audchf for values from ('2013-01-01') to ('2014-01-01')  tablespace fxarch;
create table audchf_2012 partition of audchf for values from ('2012-01-01') to ('2013-01-01')  tablespace fxarch;
create table audchf_2011 partition of audchf for values from ('2011-01-01') to ('2012-01-01')  tablespace fxarch;
create table audchf_2010 partition of audchf for values from ('2010-01-01') to ('2011-01-01')  tablespace fxarch;
create table audchfarch partition of audchf for values from ('2000-01-01') to ('2010-01-01')  tablespace fxarch;


create table audjpy ( like audjpyrmp ) partition by range (time);
create table audjpy_2018 partition of audjpy for values from ('2018-01-01') to ('2019-01-01');
create table audjpy_2017 partition of audjpy for values from ('2017-01-01') to ('2018-01-01')  tablespace fxarch;
create table audjpy_2016 partition of audjpy for values from ('2016-01-01') to ('2017-01-01')  tablespace fxarch;
create table audjpy_2015 partition of audjpy for values from ('2015-01-01') to ('2016-01-01')  tablespace fxarch;
create table audjpy_2014 partition of audjpy for values from ('2014-01-01') to ('2015-01-01')  tablespace fxarch;
create table audjpy_2013 partition of audjpy for values from ('2013-01-01') to ('2014-01-01')  tablespace fxarch;
create table audjpy_2012 partition of audjpy for values from ('2012-01-01') to ('2013-01-01')  tablespace fxarch;
create table audjpy_2011 partition of audjpy for values from ('2011-01-01') to ('2012-01-01')  tablespace fxarch;
create table audjpy_2010 partition of audjpy for values from ('2010-01-01') to ('2011-01-01')  tablespace fxarch;
create table audjpyarch partition of audjpy for values from ('2000-01-01') to ('2010-01-01')  tablespace fxarch;


create table audnzd ( like audnzdrmp ) partition by range (time);
create table audnzd_2018 partition of audnzd for values from ('2018-01-01') to ('2019-01-01');
create table audnzd_2017 partition of audnzd for values from ('2017-01-01') to ('2018-01-01')  tablespace fxarch;
create table audnzd_2016 partition of audnzd for values from ('2016-01-01') to ('2017-01-01')  tablespace fxarch;
create table audnzd_2015 partition of audnzd for values from ('2015-01-01') to ('2016-01-01')  tablespace fxarch;
create table audnzd_2014 partition of audnzd for values from ('2014-01-01') to ('2015-01-01')  tablespace fxarch;
create table audnzd_2013 partition of audnzd for values from ('2013-01-01') to ('2014-01-01')  tablespace fxarch;
create table audnzd_2012 partition of audnzd for values from ('2012-01-01') to ('2013-01-01')  tablespace fxarch;
create table audnzd_2011 partition of audnzd for values from ('2011-01-01') to ('2012-01-01')  tablespace fxarch;
create table audnzd_2010 partition of audnzd for values from ('2010-01-01') to ('2011-01-01')  tablespace fxarch;
create table audnzdarch partition of audnzd for values from ('2000-01-01') to ('2010-01-01')  tablespace fxarch;


create table audusd ( like audusdrmp ) partition by range (time);
create table audusd_2018 partition of audusd for values from ('2018-01-01') to ('2019-01-01');
create table audusd_2017 partition of audusd for values from ('2017-01-01') to ('2018-01-01')  tablespace fxarch;
create table audusd_2016 partition of audusd for values from ('2016-01-01') to ('2017-01-01')  tablespace fxarch;
create table audusd_2015 partition of audusd for values from ('2015-01-01') to ('2016-01-01')  tablespace fxarch;
create table audusd_2014 partition of audusd for values from ('2014-01-01') to ('2015-01-01')  tablespace fxarch;
create table audusd_2013 partition of audusd for values from ('2013-01-01') to ('2014-01-01')  tablespace fxarch;
create table audusd_2012 partition of audusd for values from ('2012-01-01') to ('2013-01-01')  tablespace fxarch;
create table audusd_2011 partition of audusd for values from ('2011-01-01') to ('2012-01-01')  tablespace fxarch;
create table audusd_2010 partition of audusd for values from ('2010-01-01') to ('2011-01-01')  tablespace fxarch;
create table audusdarch partition of audusd for values from ('2000-01-01') to ('2010-01-01')  tablespace fxarch;


create table cadchf ( like cadchfrmp ) partition by range (time);
create table cadchf_2018 partition of cadchf for values from ('2018-01-01') to ('2019-01-01');
create table cadchf_2017 partition of cadchf for values from ('2017-01-01') to ('2018-01-01')  tablespace fxarch;
create table cadchf_2016 partition of cadchf for values from ('2016-01-01') to ('2017-01-01')  tablespace fxarch;
create table cadchf_2015 partition of cadchf for values from ('2015-01-01') to ('2016-01-01')  tablespace fxarch;
create table cadchf_2014 partition of cadchf for values from ('2014-01-01') to ('2015-01-01')  tablespace fxarch;
create table cadchf_2013 partition of cadchf for values from ('2013-01-01') to ('2014-01-01')  tablespace fxarch;
create table cadchf_2012 partition of cadchf for values from ('2012-01-01') to ('2013-01-01')  tablespace fxarch;
create table cadchf_2011 partition of cadchf for values from ('2011-01-01') to ('2012-01-01')  tablespace fxarch;
create table cadchf_2010 partition of cadchf for values from ('2010-01-01') to ('2011-01-01')  tablespace fxarch;
create table cadchfarch partition of cadchf for values from ('2000-01-01') to ('2010-01-01')  tablespace fxarch;


create table cadjpy ( like cadjpyrmp ) partition by range (time);
create table cadjpy_2018 partition of cadjpy for values from ('2018-01-01') to ('2019-01-01');
create table cadjpy_2017 partition of cadjpy for values from ('2017-01-01') to ('2018-01-01')  tablespace fxarch;
create table cadjpy_2016 partition of cadjpy for values from ('2016-01-01') to ('2017-01-01')  tablespace fxarch;
create table cadjpy_2015 partition of cadjpy for values from ('2015-01-01') to ('2016-01-01')  tablespace fxarch;
create table cadjpy_2014 partition of cadjpy for values from ('2014-01-01') to ('2015-01-01')  tablespace fxarch;
create table cadjpy_2013 partition of cadjpy for values from ('2013-01-01') to ('2014-01-01')  tablespace fxarch;
create table cadjpy_2012 partition of cadjpy for values from ('2012-01-01') to ('2013-01-01')  tablespace fxarch;
create table cadjpy_2011 partition of cadjpy for values from ('2011-01-01') to ('2012-01-01')  tablespace fxarch;
create table cadjpy_2010 partition of cadjpy for values from ('2010-01-01') to ('2011-01-01')  tablespace fxarch;
create table cadjpyarch partition of cadjpy for values from ('2000-01-01') to ('2010-01-01')  tablespace fxarch;


create table chfjpy ( like chfjpyrmp ) partition by range (time);
create table chfjpy_2018 partition of chfjpy for values from ('2018-01-01') to ('2019-01-01');
create table chfjpy_2017 partition of chfjpy for values from ('2017-01-01') to ('2018-01-01')  tablespace fxarch;
create table chfjpy_2016 partition of chfjpy for values from ('2016-01-01') to ('2017-01-01')  tablespace fxarch;
create table chfjpy_2015 partition of chfjpy for values from ('2015-01-01') to ('2016-01-01')  tablespace fxarch;
create table chfjpy_2014 partition of chfjpy for values from ('2014-01-01') to ('2015-01-01')  tablespace fxarch;
create table chfjpy_2013 partition of chfjpy for values from ('2013-01-01') to ('2014-01-01')  tablespace fxarch;
create table chfjpy_2012 partition of chfjpy for values from ('2012-01-01') to ('2013-01-01')  tablespace fxarch;
create table chfjpy_2011 partition of chfjpy for values from ('2011-01-01') to ('2012-01-01')  tablespace fxarch;
create table chfjpy_2010 partition of chfjpy for values from ('2010-01-01') to ('2011-01-01')  tablespace fxarch;
create table chfjpyarch partition of chfjpy for values from ('2000-01-01') to ('2010-01-01')  tablespace fxarch;


create table euraud ( like euraudrmp ) partition by range (time);
create table euraud_2018 partition of euraud for values from ('2018-01-01') to ('2019-01-01');
create table euraud_2017 partition of euraud for values from ('2017-01-01') to ('2018-01-01')  tablespace fxarch;
create table euraud_2016 partition of euraud for values from ('2016-01-01') to ('2017-01-01')  tablespace fxarch;
create table euraud_2015 partition of euraud for values from ('2015-01-01') to ('2016-01-01')  tablespace fxarch;
create table euraud_2014 partition of euraud for values from ('2014-01-01') to ('2015-01-01')  tablespace fxarch;
create table euraud_2013 partition of euraud for values from ('2013-01-01') to ('2014-01-01')  tablespace fxarch;
create table euraud_2012 partition of euraud for values from ('2012-01-01') to ('2013-01-01')  tablespace fxarch;
create table euraud_2011 partition of euraud for values from ('2011-01-01') to ('2012-01-01')  tablespace fxarch;
create table euraud_2010 partition of euraud for values from ('2010-01-01') to ('2011-01-01')  tablespace fxarch;
create table euraudarch partition of euraud for values from ('2000-01-01') to ('2010-01-01')  tablespace fxarch;


create table eurcad ( like eurcadrmp ) partition by range (time);
create table eurcad_2018 partition of eurcad for values from ('2018-01-01') to ('2019-01-01');
create table eurcad_2017 partition of eurcad for values from ('2017-01-01') to ('2018-01-01')  tablespace fxarch;
create table eurcad_2016 partition of eurcad for values from ('2016-01-01') to ('2017-01-01')  tablespace fxarch;
create table eurcad_2015 partition of eurcad for values from ('2015-01-01') to ('2016-01-01')  tablespace fxarch;
create table eurcad_2014 partition of eurcad for values from ('2014-01-01') to ('2015-01-01')  tablespace fxarch;
create table eurcad_2013 partition of eurcad for values from ('2013-01-01') to ('2014-01-01')  tablespace fxarch;
create table eurcad_2012 partition of eurcad for values from ('2012-01-01') to ('2013-01-01')  tablespace fxarch;
create table eurcad_2011 partition of eurcad for values from ('2011-01-01') to ('2012-01-01')  tablespace fxarch;
create table eurcad_2010 partition of eurcad for values from ('2010-01-01') to ('2011-01-01')  tablespace fxarch;
create table eurcadarch partition of eurcad for values from ('2000-01-01') to ('2010-01-01')  tablespace fxarch;


create table eurchf ( like eurchfrmp ) partition by range (time);
create table eurchf_2018 partition of eurchf for values from ('2018-01-01') to ('2019-01-01');
create table eurchf_2017 partition of eurchf for values from ('2017-01-01') to ('2018-01-01')  tablespace fxarch;
create table eurchf_2016 partition of eurchf for values from ('2016-01-01') to ('2017-01-01')  tablespace fxarch;
create table eurchf_2015 partition of eurchf for values from ('2015-01-01') to ('2016-01-01')  tablespace fxarch;
create table eurchf_2014 partition of eurchf for values from ('2014-01-01') to ('2015-01-01')  tablespace fxarch;
create table eurchf_2013 partition of eurchf for values from ('2013-01-01') to ('2014-01-01')  tablespace fxarch;
create table eurchf_2012 partition of eurchf for values from ('2012-01-01') to ('2013-01-01')  tablespace fxarch;
create table eurchf_2011 partition of eurchf for values from ('2011-01-01') to ('2012-01-01')  tablespace fxarch;
create table eurchf_2010 partition of eurchf for values from ('2010-01-01') to ('2011-01-01')  tablespace fxarch;
create table eurchfarch partition of eurchf for values from ('2000-01-01') to ('2010-01-01')  tablespace fxarch;


create table eurgbp ( like eurgbprmp ) partition by range (time);
create table eurgbp_2018 partition of eurgbp for values from ('2018-01-01') to ('2019-01-01');
create table eurgbp_2017 partition of eurgbp for values from ('2017-01-01') to ('2018-01-01')  tablespace fxarch;
create table eurgbp_2016 partition of eurgbp for values from ('2016-01-01') to ('2017-01-01')  tablespace fxarch;
create table eurgbp_2015 partition of eurgbp for values from ('2015-01-01') to ('2016-01-01')  tablespace fxarch;
create table eurgbp_2014 partition of eurgbp for values from ('2014-01-01') to ('2015-01-01')  tablespace fxarch;
create table eurgbp_2013 partition of eurgbp for values from ('2013-01-01') to ('2014-01-01')  tablespace fxarch;
create table eurgbp_2012 partition of eurgbp for values from ('2012-01-01') to ('2013-01-01')  tablespace fxarch;
create table eurgbp_2011 partition of eurgbp for values from ('2011-01-01') to ('2012-01-01')  tablespace fxarch;
create table eurgbp_2010 partition of eurgbp for values from ('2010-01-01') to ('2011-01-01')  tablespace fxarch;
create table eurgbparch partition of eurgbp for values from ('2000-01-01') to ('2010-01-01')  tablespace fxarch;


create table eurjpy ( like eurjpyrmp ) partition by range (time);
create table eurjpy_2018 partition of eurjpy for values from ('2018-01-01') to ('2019-01-01');
create table eurjpy_2017 partition of eurjpy for values from ('2017-01-01') to ('2018-01-01')  tablespace fxarch;
create table eurjpy_2016 partition of eurjpy for values from ('2016-01-01') to ('2017-01-01')  tablespace fxarch;
create table eurjpy_2015 partition of eurjpy for values from ('2015-01-01') to ('2016-01-01')  tablespace fxarch;
create table eurjpy_2014 partition of eurjpy for values from ('2014-01-01') to ('2015-01-01')  tablespace fxarch;
create table eurjpy_2013 partition of eurjpy for values from ('2013-01-01') to ('2014-01-01')  tablespace fxarch;
create table eurjpy_2012 partition of eurjpy for values from ('2012-01-01') to ('2013-01-01')  tablespace fxarch;
create table eurjpy_2011 partition of eurjpy for values from ('2011-01-01') to ('2012-01-01')  tablespace fxarch;
create table eurjpy_2010 partition of eurjpy for values from ('2010-01-01') to ('2011-01-01')  tablespace fxarch;
create table eurjpyarch partition of eurjpy for values from ('2000-01-01') to ('2010-01-01')  tablespace fxarch;


create table eurnzd ( like eurnzdrmp ) partition by range (time);
create table eurnzd_2018 partition of eurnzd for values from ('2018-01-01') to ('2019-01-01');
create table eurnzd_2017 partition of eurnzd for values from ('2017-01-01') to ('2018-01-01')  tablespace fxarch;
create table eurnzd_2016 partition of eurnzd for values from ('2016-01-01') to ('2017-01-01')  tablespace fxarch;
create table eurnzd_2015 partition of eurnzd for values from ('2015-01-01') to ('2016-01-01')  tablespace fxarch;
create table eurnzd_2014 partition of eurnzd for values from ('2014-01-01') to ('2015-01-01')  tablespace fxarch;
create table eurnzd_2013 partition of eurnzd for values from ('2013-01-01') to ('2014-01-01')  tablespace fxarch;
create table eurnzd_2012 partition of eurnzd for values from ('2012-01-01') to ('2013-01-01')  tablespace fxarch;
create table eurnzd_2011 partition of eurnzd for values from ('2011-01-01') to ('2012-01-01')  tablespace fxarch;
create table eurnzd_2010 partition of eurnzd for values from ('2010-01-01') to ('2011-01-01')  tablespace fxarch;
create table eurnzdarch partition of eurnzd for values from ('2000-01-01') to ('2010-01-01')  tablespace fxarch;


create table eurusd ( like eurusdrmp ) partition by range (time);
create table eurusd_2018 partition of eurusd for values from ('2018-01-01') to ('2019-01-01');
create table eurusd_2017 partition of eurusd for values from ('2017-01-01') to ('2018-01-01')  tablespace fxarch;
create table eurusd_2016 partition of eurusd for values from ('2016-01-01') to ('2017-01-01')  tablespace fxarch;
create table eurusd_2015 partition of eurusd for values from ('2015-01-01') to ('2016-01-01')  tablespace fxarch;
create table eurusd_2014 partition of eurusd for values from ('2014-01-01') to ('2015-01-01')  tablespace fxarch;
create table eurusd_2013 partition of eurusd for values from ('2013-01-01') to ('2014-01-01')  tablespace fxarch;
create table eurusd_2012 partition of eurusd for values from ('2012-01-01') to ('2013-01-01')  tablespace fxarch;
create table eurusd_2011 partition of eurusd for values from ('2011-01-01') to ('2012-01-01')  tablespace fxarch;
create table eurusd_2010 partition of eurusd for values from ('2010-01-01') to ('2011-01-01')  tablespace fxarch;
create table eurusdarch partition of eurusd for values from ('2000-01-01') to ('2010-01-01')  tablespace fxarch;


create table gbpaud ( like gbpaudrmp ) partition by range (time);
create table gbpaud_2018 partition of gbpaud for values from ('2018-01-01') to ('2019-01-01');
create table gbpaud_2017 partition of gbpaud for values from ('2017-01-01') to ('2018-01-01')  tablespace fxarch;
create table gbpaud_2016 partition of gbpaud for values from ('2016-01-01') to ('2017-01-01')  tablespace fxarch;
create table gbpaud_2015 partition of gbpaud for values from ('2015-01-01') to ('2016-01-01')  tablespace fxarch;
create table gbpaud_2014 partition of gbpaud for values from ('2014-01-01') to ('2015-01-01')  tablespace fxarch;
create table gbpaud_2013 partition of gbpaud for values from ('2013-01-01') to ('2014-01-01')  tablespace fxarch;
create table gbpaud_2012 partition of gbpaud for values from ('2012-01-01') to ('2013-01-01')  tablespace fxarch;
create table gbpaud_2011 partition of gbpaud for values from ('2011-01-01') to ('2012-01-01')  tablespace fxarch;
create table gbpaud_2010 partition of gbpaud for values from ('2010-01-01') to ('2011-01-01')  tablespace fxarch;
create table gbpaudarch partition of gbpaud for values from ('2000-01-01') to ('2010-01-01')  tablespace fxarch;


create table gbpcad ( like gbpcadrmp ) partition by range (time);
create table gbpcad_2018 partition of gbpcad for values from ('2018-01-01') to ('2019-01-01');
create table gbpcad_2017 partition of gbpcad for values from ('2017-01-01') to ('2018-01-01')  tablespace fxarch;
create table gbpcad_2016 partition of gbpcad for values from ('2016-01-01') to ('2017-01-01')  tablespace fxarch;
create table gbpcad_2015 partition of gbpcad for values from ('2015-01-01') to ('2016-01-01')  tablespace fxarch;
create table gbpcad_2014 partition of gbpcad for values from ('2014-01-01') to ('2015-01-01')  tablespace fxarch;
create table gbpcad_2013 partition of gbpcad for values from ('2013-01-01') to ('2014-01-01')  tablespace fxarch;
create table gbpcad_2012 partition of gbpcad for values from ('2012-01-01') to ('2013-01-01')  tablespace fxarch;
create table gbpcad_2011 partition of gbpcad for values from ('2011-01-01') to ('2012-01-01')  tablespace fxarch;
create table gbpcad_2010 partition of gbpcad for values from ('2010-01-01') to ('2011-01-01')  tablespace fxarch;
create table gbpcadarch partition of gbpcad for values from ('2000-01-01') to ('2010-01-01')  tablespace fxarch;


create table gbpchf ( like gbpchfrmp ) partition by range (time);
create table gbpchf_2018 partition of gbpchf for values from ('2018-01-01') to ('2019-01-01');
create table gbpchf_2017 partition of gbpchf for values from ('2017-01-01') to ('2018-01-01')  tablespace fxarch;
create table gbpchf_2016 partition of gbpchf for values from ('2016-01-01') to ('2017-01-01')  tablespace fxarch;
create table gbpchf_2015 partition of gbpchf for values from ('2015-01-01') to ('2016-01-01')  tablespace fxarch;
create table gbpchf_2014 partition of gbpchf for values from ('2014-01-01') to ('2015-01-01')  tablespace fxarch;
create table gbpchf_2013 partition of gbpchf for values from ('2013-01-01') to ('2014-01-01')  tablespace fxarch;
create table gbpchf_2012 partition of gbpchf for values from ('2012-01-01') to ('2013-01-01')  tablespace fxarch;
create table gbpchf_2011 partition of gbpchf for values from ('2011-01-01') to ('2012-01-01')  tablespace fxarch;
create table gbpchf_2010 partition of gbpchf for values from ('2010-01-01') to ('2011-01-01')  tablespace fxarch;
create table gbpchfarch partition of gbpchf for values from ('2000-01-01') to ('2010-01-01')  tablespace fxarch;


create table gbpjpy ( like gbpjpyrmp ) partition by range (time);
create table gbpjpy_2018 partition of gbpjpy for values from ('2018-01-01') to ('2019-01-01');
create table gbpjpy_2017 partition of gbpjpy for values from ('2017-01-01') to ('2018-01-01')  tablespace fxarch;
create table gbpjpy_2016 partition of gbpjpy for values from ('2016-01-01') to ('2017-01-01')  tablespace fxarch;
create table gbpjpy_2015 partition of gbpjpy for values from ('2015-01-01') to ('2016-01-01')  tablespace fxarch;
create table gbpjpy_2014 partition of gbpjpy for values from ('2014-01-01') to ('2015-01-01')  tablespace fxarch;
create table gbpjpy_2013 partition of gbpjpy for values from ('2013-01-01') to ('2014-01-01')  tablespace fxarch;
create table gbpjpy_2012 partition of gbpjpy for values from ('2012-01-01') to ('2013-01-01')  tablespace fxarch;
create table gbpjpy_2011 partition of gbpjpy for values from ('2011-01-01') to ('2012-01-01')  tablespace fxarch;
create table gbpjpy_2010 partition of gbpjpy for values from ('2010-01-01') to ('2011-01-01')  tablespace fxarch;
create table gbpjpyarch partition of gbpjpy for values from ('2000-01-01') to ('2010-01-01')  tablespace fxarch;


create table gbpnzd ( like gbpnzdrmp ) partition by range (time);
create table gbpnzd_2018 partition of gbpnzd for values from ('2018-01-01') to ('2019-01-01');
create table gbpnzd_2017 partition of gbpnzd for values from ('2017-01-01') to ('2018-01-01')  tablespace fxarch;
create table gbpnzd_2016 partition of gbpnzd for values from ('2016-01-01') to ('2017-01-01')  tablespace fxarch;
create table gbpnzd_2015 partition of gbpnzd for values from ('2015-01-01') to ('2016-01-01')  tablespace fxarch;
create table gbpnzd_2014 partition of gbpnzd for values from ('2014-01-01') to ('2015-01-01')  tablespace fxarch;
create table gbpnzd_2013 partition of gbpnzd for values from ('2013-01-01') to ('2014-01-01')  tablespace fxarch;
create table gbpnzd_2012 partition of gbpnzd for values from ('2012-01-01') to ('2013-01-01')  tablespace fxarch;
create table gbpnzd_2011 partition of gbpnzd for values from ('2011-01-01') to ('2012-01-01')  tablespace fxarch;
create table gbpnzd_2010 partition of gbpnzd for values from ('2010-01-01') to ('2011-01-01')  tablespace fxarch;
create table gbpnzdarch partition of gbpnzd for values from ('2000-01-01') to ('2010-01-01')  tablespace fxarch;


create table gbpusd ( like gbpusdrmp ) partition by range (time);
create table gbpusd_2018 partition of gbpusd for values from ('2018-01-01') to ('2019-01-01');
create table gbpusd_2017 partition of gbpusd for values from ('2017-01-01') to ('2018-01-01')  tablespace fxarch;
create table gbpusd_2016 partition of gbpusd for values from ('2016-01-01') to ('2017-01-01')  tablespace fxarch;
create table gbpusd_2015 partition of gbpusd for values from ('2015-01-01') to ('2016-01-01')  tablespace fxarch;
create table gbpusd_2014 partition of gbpusd for values from ('2014-01-01') to ('2015-01-01')  tablespace fxarch;
create table gbpusd_2013 partition of gbpusd for values from ('2013-01-01') to ('2014-01-01')  tablespace fxarch;
create table gbpusd_2012 partition of gbpusd for values from ('2012-01-01') to ('2013-01-01')  tablespace fxarch;
create table gbpusd_2011 partition of gbpusd for values from ('2011-01-01') to ('2012-01-01')  tablespace fxarch;
create table gbpusd_2010 partition of gbpusd for values from ('2010-01-01') to ('2011-01-01')  tablespace fxarch;
create table gbpusdarch partition of gbpusd for values from ('2000-01-01') to ('2010-01-01')  tablespace fxarch;


create table nzdcad ( like nzdcadrmp ) partition by range (time);
create table nzdcad_2018 partition of nzdcad for values from ('2018-01-01') to ('2019-01-01');
create table nzdcad_2017 partition of nzdcad for values from ('2017-01-01') to ('2018-01-01')  tablespace fxarch;
create table nzdcad_2016 partition of nzdcad for values from ('2016-01-01') to ('2017-01-01')  tablespace fxarch;
create table nzdcad_2015 partition of nzdcad for values from ('2015-01-01') to ('2016-01-01')  tablespace fxarch;
create table nzdcad_2014 partition of nzdcad for values from ('2014-01-01') to ('2015-01-01')  tablespace fxarch;
create table nzdcad_2013 partition of nzdcad for values from ('2013-01-01') to ('2014-01-01')  tablespace fxarch;
create table nzdcad_2012 partition of nzdcad for values from ('2012-01-01') to ('2013-01-01')  tablespace fxarch;
create table nzdcad_2011 partition of nzdcad for values from ('2011-01-01') to ('2012-01-01')  tablespace fxarch;
create table nzdcad_2010 partition of nzdcad for values from ('2010-01-01') to ('2011-01-01')  tablespace fxarch;
create table nzdcadarch partition of nzdcad for values from ('2000-01-01') to ('2010-01-01')  tablespace fxarch;


create table nzdchf ( like nzdchfrmp ) partition by range (time);
create table nzdchf_2018 partition of nzdchf for values from ('2018-01-01') to ('2019-01-01');
create table nzdchf_2017 partition of nzdchf for values from ('2017-01-01') to ('2018-01-01')  tablespace fxarch;
create table nzdchf_2016 partition of nzdchf for values from ('2016-01-01') to ('2017-01-01')  tablespace fxarch;
create table nzdchf_2015 partition of nzdchf for values from ('2015-01-01') to ('2016-01-01')  tablespace fxarch;
create table nzdchf_2014 partition of nzdchf for values from ('2014-01-01') to ('2015-01-01')  tablespace fxarch;
create table nzdchf_2013 partition of nzdchf for values from ('2013-01-01') to ('2014-01-01')  tablespace fxarch;
create table nzdchf_2012 partition of nzdchf for values from ('2012-01-01') to ('2013-01-01')  tablespace fxarch;
create table nzdchf_2011 partition of nzdchf for values from ('2011-01-01') to ('2012-01-01')  tablespace fxarch;
create table nzdchf_2010 partition of nzdchf for values from ('2010-01-01') to ('2011-01-01')  tablespace fxarch;
create table nzdchfarch partition of nzdchf for values from ('2000-01-01') to ('2010-01-01')  tablespace fxarch;


create table nzdjpy ( like nzdjpyrmp ) partition by range (time);
create table nzdjpy_2018 partition of nzdjpy for values from ('2018-01-01') to ('2019-01-01');
create table nzdjpy_2017 partition of nzdjpy for values from ('2017-01-01') to ('2018-01-01')  tablespace fxarch;
create table nzdjpy_2016 partition of nzdjpy for values from ('2016-01-01') to ('2017-01-01')  tablespace fxarch;
create table nzdjpy_2015 partition of nzdjpy for values from ('2015-01-01') to ('2016-01-01')  tablespace fxarch;
create table nzdjpy_2014 partition of nzdjpy for values from ('2014-01-01') to ('2015-01-01')  tablespace fxarch;
create table nzdjpy_2013 partition of nzdjpy for values from ('2013-01-01') to ('2014-01-01')  tablespace fxarch;
create table nzdjpy_2012 partition of nzdjpy for values from ('2012-01-01') to ('2013-01-01')  tablespace fxarch;
create table nzdjpy_2011 partition of nzdjpy for values from ('2011-01-01') to ('2012-01-01')  tablespace fxarch;
create table nzdjpy_2010 partition of nzdjpy for values from ('2010-01-01') to ('2011-01-01')  tablespace fxarch;
create table nzdjpyarch partition of nzdjpy for values from ('2000-01-01') to ('2010-01-01')  tablespace fxarch;


create table nzdusd ( like nzdusdrmp ) partition by range (time);
create table nzdusd_2018 partition of nzdusd for values from ('2018-01-01') to ('2019-01-01');
create table nzdusd_2017 partition of nzdusd for values from ('2017-01-01') to ('2018-01-01')  tablespace fxarch;
create table nzdusd_2016 partition of nzdusd for values from ('2016-01-01') to ('2017-01-01')  tablespace fxarch;
create table nzdusd_2015 partition of nzdusd for values from ('2015-01-01') to ('2016-01-01')  tablespace fxarch;
create table nzdusd_2014 partition of nzdusd for values from ('2014-01-01') to ('2015-01-01')  tablespace fxarch;
create table nzdusd_2013 partition of nzdusd for values from ('2013-01-01') to ('2014-01-01')  tablespace fxarch;
create table nzdusd_2012 partition of nzdusd for values from ('2012-01-01') to ('2013-01-01')  tablespace fxarch;
create table nzdusd_2011 partition of nzdusd for values from ('2011-01-01') to ('2012-01-01')  tablespace fxarch;
create table nzdusd_2010 partition of nzdusd for values from ('2010-01-01') to ('2011-01-01')  tablespace fxarch;
create table nzdusdarch partition of nzdusd for values from ('2000-01-01') to ('2010-01-01')  tablespace fxarch;


create table usdcad ( like usdcadrmp ) partition by range (time);
create table usdcad_2018 partition of usdcad for values from ('2018-01-01') to ('2019-01-01');
create table usdcad_2017 partition of usdcad for values from ('2017-01-01') to ('2018-01-01')  tablespace fxarch;
create table usdcad_2016 partition of usdcad for values from ('2016-01-01') to ('2017-01-01')  tablespace fxarch;
create table usdcad_2015 partition of usdcad for values from ('2015-01-01') to ('2016-01-01')  tablespace fxarch;
create table usdcad_2014 partition of usdcad for values from ('2014-01-01') to ('2015-01-01')  tablespace fxarch;
create table usdcad_2013 partition of usdcad for values from ('2013-01-01') to ('2014-01-01')  tablespace fxarch;
create table usdcad_2012 partition of usdcad for values from ('2012-01-01') to ('2013-01-01')  tablespace fxarch;
create table usdcad_2011 partition of usdcad for values from ('2011-01-01') to ('2012-01-01')  tablespace fxarch;
create table usdcad_2010 partition of usdcad for values from ('2010-01-01') to ('2011-01-01')  tablespace fxarch;
create table usdcadarch partition of usdcad for values from ('2000-01-01') to ('2010-01-01')  tablespace fxarch;


create table usdchf ( like usdchfrmp ) partition by range (time);
create table usdchf_2018 partition of usdchf for values from ('2018-01-01') to ('2019-01-01');
create table usdchf_2017 partition of usdchf for values from ('2017-01-01') to ('2018-01-01')  tablespace fxarch;
create table usdchf_2016 partition of usdchf for values from ('2016-01-01') to ('2017-01-01')  tablespace fxarch;
create table usdchf_2015 partition of usdchf for values from ('2015-01-01') to ('2016-01-01')  tablespace fxarch;
create table usdchf_2014 partition of usdchf for values from ('2014-01-01') to ('2015-01-01')  tablespace fxarch;
create table usdchf_2013 partition of usdchf for values from ('2013-01-01') to ('2014-01-01')  tablespace fxarch;
create table usdchf_2012 partition of usdchf for values from ('2012-01-01') to ('2013-01-01')  tablespace fxarch;
create table usdchf_2011 partition of usdchf for values from ('2011-01-01') to ('2012-01-01')  tablespace fxarch;
create table usdchf_2010 partition of usdchf for values from ('2010-01-01') to ('2011-01-01')  tablespace fxarch;
create table usdchfarch partition of usdchf for values from ('2000-01-01') to ('2010-01-01')  tablespace fxarch;


create table usdjpy ( like usdjpyrmp ) partition by range (time);
create table usdjpy_2018 partition of usdjpy for values from ('2018-01-01') to ('2019-01-01');
create table usdjpy_2017 partition of usdjpy for values from ('2017-01-01') to ('2018-01-01')  tablespace fxarch;
create table usdjpy_2016 partition of usdjpy for values from ('2016-01-01') to ('2017-01-01')  tablespace fxarch;
create table usdjpy_2015 partition of usdjpy for values from ('2015-01-01') to ('2016-01-01')  tablespace fxarch;
create table usdjpy_2014 partition of usdjpy for values from ('2014-01-01') to ('2015-01-01')  tablespace fxarch;
create table usdjpy_2013 partition of usdjpy for values from ('2013-01-01') to ('2014-01-01')  tablespace fxarch;
create table usdjpy_2012 partition of usdjpy for values from ('2012-01-01') to ('2013-01-01')  tablespace fxarch;
create table usdjpy_2011 partition of usdjpy for values from ('2011-01-01') to ('2012-01-01')  tablespace fxarch;
create table usdjpy_2010 partition of usdjpy for values from ('2010-01-01') to ('2011-01-01')  tablespace fxarch;
create table usdjpyarch partition of usdjpy for values from ('2000-01-01') to ('2010-01-01')  tablespace fxarch;

#insert into audcad select * from audcadrmp;
#insert into audchf select * from audchfrmp;
#insert into audjpy select * from audjpyrmp;
#insert into audnzd select * from audnzdrmp;
#insert into audusd select * from audusdrmp;
#insert into cadchf select * from cadchfrmp;
#insert into cadjpy select * from cadjpyrmp;
#insert into chfjpy select * from chfjpyrmp;
#insert into euraud select * from euraudrmp;
#insert into eurcad select * from eurcadrmp;
#insert into eurchf select * from eurchfrmp;
#insert into eurgbp select * from eurgbprmp;
#insert into eurjpy select * from eurjpyrmp;
#insert into eurnzd select * from eurnzdrmp;
#insert into eurusd select * from eurusdrmp;
#insert into gbpaud select * from gbpaudrmp;
#insert into gbpcad select * from gbpcadrmp;
#insert into gbpchf select * from gbpchfrmp;
#insert into gbpjpy select * from gbpjpyrmp;
#insert into gbpnzd select * from gbpnzdrmp;
#insert into gbpusd select * from gbpusdrmp;
#insert into nzdcad select * from nzdcadrmp;
#insert into nzdchf select * from nzdchfrmp;
#insert into nzdjpy select * from nzdjpyrmp;
#insert into nzdusd select * from nzdusdrmp;
#insert into usdcad select * from usdcadrmp;
#insert into usdchf select * from usdchfrmp;
#insert into usdjpy select * from usdjpyrmp;
