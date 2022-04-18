import Vue from 'vue'
import App from './App.vue'
import router from './router'
import vuetify from './plugins/vuetify';
import GeoLocation from 'vue-browser-geolocation'
// import firebase from 'firebase'


Vue.config.productionTip = false
Vue.config.silent = true
Vue.use(GeoLocation)



new Vue({
  data:{
    authenticated: false,
    username:'',
    accidents:[],
    cred:''
  },
  router,
  vuetify,
  render: h => h(App)
}).$mount('#app')
