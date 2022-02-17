node {
    def app

    stage('Clone repository') {
        /* Cloning the Repository to our Workspace */

        checkout scm
    }
    stage('Build image') {
        /* This builds the actual image */

        app = docker.build("noetic")
    }
    stage('Test image') {
        
        app.inside {
            echo "Tests passed"
            sh 'py.test --junit-xml test-reports/results.xml test/kdl_parser.py'
        }
    }
     stage ('Email Notification'){
         mail bcc: '', body: 'Thanks', cc: '', from: '', replyTo: '', subject: 'Jenkinsjob Successful', to: 'alok.natheee@gmail.com'
     }
}
   
