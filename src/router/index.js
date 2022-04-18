import Vue from 'vue'
import Router from 'vue-router'
import Dash from '@/components/home/Dash'
import Welcome from '@/components/home/Welcome'
import Signup from '@/components/auth/Signup'
import VueChatScroll from 'vue-chat-scroll'
import Login from '@/components/auth/Login'
// import HomePage from '@/components/home/HomePage'
import AccidentHistory from '@/components/home/AccidentHistory'
import firebase from 'firebase'
import Settings from '@/components/home/Settings'



Vue.use(VueChatScroll)

Vue.use(Router)

const router = new Router({
  mode: 'history',
  routes: [
    {
      path: '/dash',
      name: 'Dash',
      component: Dash,
      props: true,
      meta:{
        requiresAuth: true
      }
      
    },
    {
      path: '/',
      name: 'Welcome',
      component: Welcome,
      
    },
    {
      path: '/history',
      name: 'AccidentHistory',
      component: AccidentHistory,
      props: true,
      meta:{
        requiresAuth: true
      }
      
    },
    {
      path:'/signup',
      name:'Signup',
      component: Signup
    },
    {
      path:'/login',
      name:'Login',
      component: Login
    },
    {
      path:'/settings',
      name:'Settings',
      component: Settings,
      meta:{
        requiresAuth: true
      }
    }
  ]
})

router.beforeEach((to,from,next)=>{
  //check to see if route requires auth
  if(to.matched.some(rec=>rec.meta.requiresAuth)){
    //check auth state of user
    let user = firebase.auth().currentUser
    if(user){
      //yes
      
      next()
    }else{
      //not
      next({name:'Login'})
    }
  }else{
    next()
  }
})


export default router